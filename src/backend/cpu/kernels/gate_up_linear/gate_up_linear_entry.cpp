#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/packed_weight_validation.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/check.h"
#include "aethermind/operators/op_params.h"
#include "gate_up_linear_internal.h"
#include "utils/overflow_check.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace aethermind::cpu::detail {
namespace {

StatusOr<int64_t> TotalOutFeatures(const GateUpLinearF32KernelMetadata& metadata) noexcept {
    if (metadata.gate_out_features < 0 || metadata.up_out_features < 0) {
        return Status::InvalidArgument(
                "GateUpLinearKernelEntry requires non-negative output features");
    }

    int64_t total_features = 0;
    if (CheckOverflowAdd(metadata.gate_out_features,
                         metadata.up_out_features,
                         &total_features)) {
        return Status::Overflow(
                "GateUpLinearKernelEntry output feature count overflows int64_t");
    }
    return total_features;
}

StatusOr<GateUpLinearF32KernelMetadata> ReadMetadata(
        std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(GateUpLinearF32KernelMetadata)) {
        return Status::InvalidArgument("GateUpLinearKernelEntry requires "
                                       "Gate-Up metadata in attrs");
    }

    GateUpLinearF32KernelMetadata metadata{};
    std::memcpy(&metadata, attrs.data(), sizeof(metadata));
    AM_RETURN_IF_ERROR(TotalOutFeatures(metadata).status());
    return metadata;
}

Status BuildGateUpLinearF32IdentityArgs(const KernelParamsBuildContext& context,
                                        void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 2U) {
        return Status::InvalidArgument("GateUpLinear requires one activation "
                                       "input and two outputs");
    }

    if (!context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "GateUpLinear requires a packed weight artifact");
    }

    AM_ASSIGN_OR_RETURN(const GateUpLinearF32KernelMetadata metadata,
                        ReadMetadata(context.attrs));
    AM_ASSIGN_OR_RETURN(const int64_t total_out_features,
                        TotalOutFeatures(metadata));
    const TensorView& input = context.inputs[0];
    const MutableTensorView& gate = context.outputs[0];
    const MutableTensorView& up = context.outputs[1];

    if (!input.is_valid() || !gate.is_valid() || !up.is_valid()) {
        return Status::InvalidArgument(
                "GateUpLinearKernelEntry requires valid TensorViews");
    }

    if (input.dtype() != DataType::Float32() || gate.dtype() != DataType::Float32() ||
        up.dtype() != DataType::Float32()) {
        return Status::InvalidArgument("GateUpLinearKernelEntry requires "
                                       "float32 input and outputs");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || gate.rank() != rank || up.rank() != rank) {
        return Status::InvalidArgument(
                "GateUpLinearKernelEntry requires input and outputs "
                "with matching rank >= 1");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (gate.dim(dim) != input.dim(dim) || up.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "GateUpLinearKernelEntry requires matching leading dimensions");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    if (gate.dim(rank - 1) != metadata.gate_out_features ||
        up.dim(rank - 1) != metadata.up_out_features) {
        return Status::InvalidArgument(
                "GateUpLinearKernelEntry outputs do not match Gate-Up feature metadata");
    }

    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            *context.packed_weight, std::array{total_out_features, in_features},
            "GateUpLinearKernelEntry"));

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "GateUpLinearKernelEntry"));

    if (row_count == 0) {
        ::new (params_buffer) GateUpLinearA32W32KernelArgs{
                .row_count = 0,
                .in_features = in_features,
                .gate_out_features = metadata.gate_out_features,
                .up_out_features = metadata.up_out_features,
        };
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis gate_analysis,
                        AnalyzeRowwiseView(gate, "GateUpLinearKernelEntry gate output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis up_analysis,
                        AnalyzeRowwiseView(up, "GateUpLinearKernelEntry up output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "GateUpLinearKernelEntry gate output", gate_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "GateUpLinearKernelEntry up output", up_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinear", gate_analysis.footprint(), "gate output",
            up_analysis.footprint(), "up output"));

    GateUpLinearA32W32KernelArgs args{
            .gate = gate.data<float>(),
            .up = up.data<float>(),
            .row_count = row_count,
            .in_features = in_features,
            .gate_out_features = metadata.gate_out_features,
            .up_out_features = metadata.up_out_features,
            .gate_row_stride = gate_analysis.row_stride(),
            .gate_col_stride = gate_analysis.column_stride(),
            .up_row_stride = up_analysis.row_stride(),
            .up_col_stride = up_analysis.column_stride(),
    };

    if (total_out_features == 0) {
        ::new (params_buffer) GateUpLinearA32W32KernelArgs(args);
        return Status::Ok();
    }

    if ((metadata.gate_out_features != 0 && gate.data() == nullptr) ||
        (metadata.up_out_features != 0 && up.data() == nullptr) ||
        (in_features != 0 &&
         (input.data() == nullptr || context.packed_weight->data == nullptr))) {
        return Status::InvalidArgument(
                "GateUpLinearKernelEntry requires non-null data for non-empty tensors");
    }

    if (in_features == 0) {
        ::new (params_buffer) GateUpLinearA32W32KernelArgs(args);
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "GateUpLinearKernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinear", gate_analysis.footprint(), "gate output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinear", up_analysis.footprint(), "up output",
            input_analysis.footprint(), "input"));

    if (!std::in_range<int64_t>(context.packed_weight->nbytes)) {
        return Status::Overflow(
                "GateUpLinearKernelEntry packed storage size exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(
            const ByteAddressRange packed_range,
            BuildContiguousByteRange(context.packed_weight->data,
                                     static_cast<int64_t>(context.packed_weight->nbytes),
                                     size_t{1}, "GateUpLinearKernelEntry packed weight"));
    for (const auto& [footprint, role]: std::array{
                 std::pair{gate_analysis.footprint(), "gate output"},
                 std::pair{up_analysis.footprint(), "up output"}}) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU GateUpLinear", footprint, role, packed_range,
                "packed weight"));
    }

    int64_t up_weight_offset = 0;
    if (CheckOverflowMul(metadata.gate_out_features, in_features, &up_weight_offset)) {
        return Status::Overflow(
                "GateUpLinearKernelEntry packed weight offset overflows int64_t");
    }

    args.input = input.data<float>();
    const auto* packed_data = static_cast<const float*>(context.packed_weight->data);
    args.gate_weight = packed_data;
    args.up_weight = packed_data + up_weight_offset;
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    ::new (params_buffer) GateUpLinearA32W32KernelArgs(args);
    return Status::Ok();
}

template<bool Bfloat>
Status BuildGateUpLinearF32W16ReferenceArgs(const KernelParamsBuildContext& context,
                                            void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 2U) {
        return Status::InvalidArgument("GateUpLinearW16 requires one activation "
                                       "input and two outputs");
    }

    if (!context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "GateUpLinearW16 requires a packed weight artifact");
    }

    AM_ASSIGN_OR_RETURN(const GateUpLinearF32KernelMetadata metadata,
                        ReadMetadata(context.attrs));
    AM_ASSIGN_OR_RETURN(const int64_t total_out_features,
                        TotalOutFeatures(metadata));
    const TensorView& input = context.inputs[0];
    const MutableTensorView& gate = context.outputs[0];
    const MutableTensorView& up = context.outputs[1];

    if (!input.is_valid() || !gate.is_valid() || !up.is_valid()) {
        return Status::InvalidArgument(
                "GateUpLinearW16KernelEntry requires valid TensorViews");
    }

    if (input.dtype() != DataType::Float32() || gate.dtype() != DataType::Float32() ||
        up.dtype() != DataType::Float32()) {
        return Status::InvalidArgument("GateUpLinearW16KernelEntry requires "
                                       "float32 input and outputs");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || gate.rank() != rank || up.rank() != rank) {
        return Status::InvalidArgument(
                "GateUpLinearW16KernelEntry requires input and outputs "
                "with matching rank >= 1");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (gate.dim(dim) != input.dim(dim) || up.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "GateUpLinearW16KernelEntry requires matching leading dimensions");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    if (gate.dim(rank - 1) != metadata.gate_out_features ||
        up.dim(rank - 1) != metadata.up_out_features) {
        return Status::InvalidArgument(
                "GateUpLinearW16KernelEntry outputs do "
                "not match Gate-Up feature metadata");
    }

    const DataType weight_dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            *context.packed_weight, std::array{total_out_features, in_features},
            "GateUpLinearW16KernelEntry", weight_dtype));

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "GateUpLinearW16KernelEntry"));
    if (row_count == 0) {
        ::new (params_buffer) GateUpLinearA32W16KernelArgs{
                .weight_dtype = weight_dtype,
                .row_count = 0,
                .in_features = in_features,
                .gate_out_features = metadata.gate_out_features,
                .up_out_features = metadata.up_out_features,
        };
        return Status::Ok();
    }

    if ((metadata.gate_out_features != 0 && reinterpret_cast<std::uintptr_t>(gate.data()) % alignof(float) != 0) ||
        (metadata.up_out_features != 0 && reinterpret_cast<std::uintptr_t>(up.data()) % alignof(float) != 0) ||
        (total_out_features != 0 && in_features != 0 && reinterpret_cast<std::uintptr_t>(input.data()) % alignof(float) != 0)) {
        return Status::InvalidArgument(
                "GateUpLinearW16 requires naturally aligned float32 input/output data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis gate_analysis,
                        AnalyzeRowwiseView(gate, "GateUpLinearW16KernelEntry gate output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis up_analysis,
                        AnalyzeRowwiseView(up, "GateUpLinearW16KernelEntry up output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "GateUpLinearW16KernelEntry gate output", gate_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "GateUpLinearW16KernelEntry up output", up_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinearW16", gate_analysis.footprint(), "gate output",
            up_analysis.footprint(), "up output"));

    GateUpLinearA32W16KernelArgs args{
            .weight_dtype = weight_dtype,
            .gate = gate.data<float>(),
            .up = up.data<float>(),
            .row_count = row_count,
            .in_features = in_features,
            .gate_out_features = metadata.gate_out_features,
            .up_out_features = metadata.up_out_features,
            .gate_row_stride = gate_analysis.row_stride(),
            .gate_col_stride = gate_analysis.column_stride(),
            .up_row_stride = up_analysis.row_stride(),
            .up_col_stride = up_analysis.column_stride(),
    };

    if (total_out_features == 0) {
        ::new (params_buffer) GateUpLinearA32W16KernelArgs(args);
        return Status::Ok();
    }

    if ((metadata.gate_out_features != 0 && gate.data() == nullptr) ||
        (metadata.up_out_features != 0 && up.data() == nullptr) ||
        (in_features != 0 &&
         (input.data() == nullptr || context.packed_weight->data == nullptr))) {
        return Status::InvalidArgument(
                "GateUpLinearW16KernelEntry requires non-null data for non-empty tensors");
    }

    if (in_features == 0) {
        ::new (params_buffer) GateUpLinearA32W16KernelArgs(args);
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "GateUpLinearW16KernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinearW16", gate_analysis.footprint(),
            "gate output", input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinearW16", up_analysis.footprint(),
            "up output", input_analysis.footprint(), "input"));

    if (!std::in_range<int64_t>(context.packed_weight->nbytes)) {
        return Status::Overflow(
                "GateUpLinearW16KernelEntry packed storage size exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(
            const ByteAddressRange packed_range,
            BuildContiguousByteRange(context.packed_weight->data,
                                     static_cast<int64_t>(context.packed_weight->nbytes),
                                     size_t{1}, "GateUpLinearW16KernelEntry packed weight"));
    for (const auto& [footprint, role]: std::array{
                 std::pair{gate_analysis.footprint(), "gate output"},
                 std::pair{up_analysis.footprint(), "up output"}}) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU GateUpLinearW16", footprint, role, packed_range,
                "packed weight"));
    }

    int64_t up_weight_offset = 0;
    if (CheckOverflowMul(metadata.gate_out_features, in_features, &up_weight_offset)) {
        return Status::Overflow(
                "GateUpLinearW16KernelEntry packed weight offset overflows int64_t");
    }
    size_t up_weight_bytes = 0;
    if (CheckOverflowMul(static_cast<size_t>(up_weight_offset), sizeof(uint16_t),
                         &up_weight_bytes)) {
        return Status::Overflow(
                "GateUpLinearW16KernelEntry packed weight byte offset overflows size_t");
    }

    args.input = input.data<float>();
    const auto* packed_data = static_cast<const std::byte*>(context.packed_weight->data);
    args.gate_weight = packed_data;
    args.up_weight = packed_data + up_weight_bytes;
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    ::new (params_buffer) GateUpLinearA32W16KernelArgs(args);
    return Status::Ok();
}

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status BuildGateUpLinearF32BpanelArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 2U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed GateUpLinear requires one activation, "
                "two outputs, and an artifact");
    }

    AM_ASSIGN_OR_RETURN(const GateUpLinearF32KernelMetadata metadata,
                        ReadMetadata(context.attrs));
    AM_ASSIGN_OR_RETURN(const int64_t total_n, TotalOutFeatures(metadata));
    const TensorView& input = context.inputs[0];
    const MutableTensorView& gate = context.outputs[0];
    const MutableTensorView& up = context.outputs[1];
    if (!input.is_valid() || !gate.is_valid() || !up.is_valid() ||
        input.dtype() != DataType::Float32() || gate.dtype() != DataType::Float32() ||
        up.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "Packed GateUpLinear requires valid float32 activation and outputs");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || gate.rank() != rank || up.rank() != rank) {
        return Status::InvalidArgument(
                "Packed GateUpLinear requires matching input/output rank >= 1");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (gate.dim(dim) != input.dim(dim) || up.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "Packed GateUpLinear requires matching leading dimensions");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    if (gate.dim(rank - 1) != metadata.gate_out_features ||
        up.dim(rank - 1) != metadata.up_out_features) {
        return Status::InvalidArgument(
                "Packed GateUpLinear outputs do not match metadata");
    }

    AM_RETURN_IF_ERROR(ValidateBPanelF32PackedWeight(
            *context.packed_weight, std::array{total_n, in_features},
            "GateUpLinearKernelEntry"));
    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "GateUpLinearKernelEntry"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis gate_analysis,
                        AnalyzeRowwiseView(gate, "GateUpLinearKernelEntry gate output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis up_analysis,
                        AnalyzeRowwiseView(up, "GateUpLinearKernelEntry up output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "GateUpLinearKernelEntry gate output", gate_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "GateUpLinearKernelEntry up output", up_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinear", gate_analysis.footprint(), "gate output",
            up_analysis.footprint(), "up output"));

    const int64_t n_blocks = total_n / kCpuBPanelF32Kc512Nr16NR +
                             (total_n % kCpuBPanelF32Kc512Nr16NR != 0);
    const auto* const packed_data = static_cast<const std::byte*>(context.packed_weight->data);
    const auto build_gemm = [&](const MutableTensorView& output,
                                const RowwiseViewAnalysis& output_analysis,
                                int64_t output_n,
                                int64_t n_offset) {
        return PackedGemmF32Args{
                .gemm = GemmArgs{
                        .lhs = static_cast<const std::byte*>(input.data()),
                        .output = static_cast<std::byte*>(output.data()),
                        .m = row_count,
                        .n = output_n,
                        .k = in_features,
                        .lhs_m_stride = 0,
                        .lhs_k_stride = 1,
                        .output_m_stride = output_analysis.row_stride(),
                        .output_n_stride = output_analysis.column_stride(),
                },
                .packed_b = packed_data,
                .packed_nbytes = context.packed_weight->nbytes,
                .logical_n = total_n,
                .logical_k = in_features,
                .weight_n_offset = n_offset,
                .n_blocks = n_blocks,
        };
    };

    GateUpLinearA32W32PackedBKernelArgs args{
            .gate = build_gemm(gate, gate_analysis,
                               metadata.gate_out_features, 0),
            .up = build_gemm(up, up_analysis,
                             metadata.up_out_features,
                             metadata.gate_out_features),
    };

    if (row_count == 0 || total_n == 0 || in_features == 0) {
        ::new (params_buffer) GateUpLinearA32W32PackedBKernelArgs(args);
        return Status::Ok();
    }

    if (input.data() == nullptr ||
        (metadata.gate_out_features != 0 && gate.data() == nullptr) ||
        (metadata.up_out_features != 0 && up.data() == nullptr)) {
        return Status::InvalidArgument(
                "Packed GateUpLinear requires non-null data for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "GateUpLinearKernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinear", gate_analysis.footprint(), "gate output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU GateUpLinear", up_analysis.footprint(), "up output",
            input_analysis.footprint(), "input"));
    if (!std::in_range<int64_t>(context.packed_weight->nbytes)) {
        return Status::Overflow(
                "GateUpLinearKernelEntry packed storage size exceeds int64_t");
    }

    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                context.packed_weight->data,
                                static_cast<int64_t>(context.packed_weight->nbytes),
                                size_t{1}, "GateUpLinearKernelEntry packed weight"));
    for (const auto& [footprint, role]: std::array{
                 std::pair{gate_analysis.footprint(), "gate output"},
                 std::pair{up_analysis.footprint(), "up output"}}) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU GateUpLinear", footprint, role, packed_range,
                "packed weight"));
    }
    args.gate.gemm.lhs_m_stride = input_analysis.row_stride();
    args.up.gemm.lhs_m_stride = input_analysis.row_stride();
    args.gate.gemm.lhs_k_stride = input_analysis.column_stride();
    args.up.gemm.lhs_k_stride = input_analysis.column_stride();
    ::new (params_buffer) GateUpLinearA32W32PackedBKernelArgs(args);
    return Status::Ok();
}

Status GateUpLinearF32PackedBEntry(const KernelContext& context) noexcept {
    const auto* args = static_cast<const GateUpLinearA32W32PackedBKernelArgs*>(
            context.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32PackedB(*args);
}
#endif

Status BuildGateUpLinearF32Metadata(const OpParams& params,
                                    std::vector<std::byte>& attrs) {
    const auto* gate_up_params = std::get_if<GateUpLinearParams>(&params);
    if (gate_up_params == nullptr || gate_up_params->has_bias) {
        return Status::InvalidArgument(
                "CPU GateUpLinear kernel requires no-bias GateUpLinearParams");
    }

    const GateUpLinearF32KernelMetadata metadata{
            .gate_out_features = gate_up_params->gate_out_features,
            .up_out_features = gate_up_params->up_out_features,
    };

    AM_RETURN_IF_ERROR(TotalOutFeatures(metadata).status());
    const auto bytes = std::as_bytes(std::span{&metadata, size_t{1}});
    attrs.assign(bytes.begin(), bytes.end());
    return Status::Ok();
}

Status GateUpLinearF32ScalarEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GateUpLinearA32W32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32Scalar(*args);
}

Status GateUpLinearF32W16ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GateUpLinearA32W16KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32W16Reference(*args);
}

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGateUpLinearF32PackedB(
        const GateUpLinearA32W32PackedBKernelArgs& args) noexcept {
    AM_RETURN_IF_ERROR(RunGemmA32W32PackedB(args.gate));
    return RunGemmA32W32PackedB(args.up);
}
#endif

static_assert(std::is_trivially_copyable_v<GateUpLinearF32KernelMetadata>);
static_assert(std::is_trivially_destructible_v<GateUpLinearA32W32KernelArgs>);
static_assert(sizeof(GateUpLinearA32W32KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(GateUpLinearA32W32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuGateUpLinearF32Scalar,
        KernelDef{
                .op_type = OpType::kGateUpLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &GateUpLinearF32ScalarEntry,
                .priority = 10,
                .params_size = sizeof(GateUpLinearA32W32KernelArgs),
                .params_builder = &BuildGateUpLinearF32IdentityArgs,
                .metadata_builder = &BuildGateUpLinearF32Metadata,
                .name = "cpu::gate_up_linear_f32_w32_scalar"})

static_assert(std::is_trivially_destructible_v<GateUpLinearA32W16KernelArgs>);
static_assert(sizeof(GateUpLinearA32W16KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(GateUpLinearA32W16KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuGateUpLinearF32F16Reference,
        KernelDef{
                .op_type = OpType::kGateUpLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &GateUpLinearF32W16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GateUpLinearA32W16KernelArgs),
                .params_builder = &BuildGateUpLinearF32W16ReferenceArgs<false>,
                .metadata_builder = &BuildGateUpLinearF32Metadata,
                .name = "cpu::gate_up_linear_f32_wf16_reference"})

AM_REGISTER_KERNEL(
        CpuGateUpLinearF32Bf16Reference,
        KernelDef{
                .op_type = OpType::kGateUpLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::BFloat(16),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &GateUpLinearF32W16ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GateUpLinearA32W16KernelArgs),
                .params_builder = &BuildGateUpLinearF32W16ReferenceArgs<true>,
                .metadata_builder = &BuildGateUpLinearF32Metadata,
                .name = "cpu::gate_up_linear_f32_wbf16_reference"})


#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
static_assert(std::is_trivially_destructible_v<GateUpLinearA32W32PackedBKernelArgs>);
static_assert(sizeof(GateUpLinearA32W32PackedBKernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(GateUpLinearA32W32PackedBKernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuGateUpLinearF32PackedBpanelCandidate,
        KernelDef{
                .op_type = OpType::kGateUpLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = cpu::CpuBPanelF32Kc512Nr16Recipe(),
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &GateUpLinearF32PackedBEntry,
                .priority = 10,
                .params_size = sizeof(GateUpLinearA32W32PackedBKernelArgs),
                .params_builder = &BuildGateUpLinearF32BpanelArgs,
                .metadata_builder = &BuildGateUpLinearF32Metadata,
                .name = "cpu::gate_up_linear_f32_w32_packed_bpanel_candidate"})
#endif


namespace {

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status GateUpLinearF32Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GateUpLinearA32W32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32Avx2Fma(*args);
}
#endif
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status GateUpLinearF32Bf16Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GateUpLinearA32W16KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32Bf16Avx2Fma(*args);
}
#endif
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
Status GateUpLinearF32F16Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GateUpLinearA32W16KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32F16Avx2Fma(*args);
}
#endif

template<bool Reduced>
KernelDef GateUpLinearOptimizedDef(DataType weight_dtype, KernelFunc entry,
                                   KernelParamsBuilder builder, CpuFeatureSet requirements,
                                   std::string_view name) {
    return KernelDef{
            .op_type = OpType::kGateUpLinear,
            .selector = KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .output_dtype = DataType::Float32(),
                    .weight_dtype = weight_dtype,
                    .weight_format = WeightFormat::kPacked,
                    .phase = ExecPhase::kBoth,
            },
            .packing_recipe = CpuIdentityPackingRecipe(),
            .cpu_requirements = requirements,
            .kernel_func = entry,
            .priority = 20,
            .params_size = Reduced ? sizeof(GateUpLinearA32W16KernelArgs) : sizeof(GateUpLinearA32W32KernelArgs),
            .params_builder = builder,
            .metadata_builder = &BuildGateUpLinearF32Metadata,
            .name = name,
    };
}

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuGateUpLinearF32Avx2Optimized,
        (GateUpLinearOptimizedDef<false>(DataType::Float32(), &GateUpLinearF32Avx2FmaEntry,
                                         &BuildGateUpLinearF32IdentityArgs, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                         "cpu::gate_up_linear_f32_w32_avx2_fma")))
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuGateUpLinearBf16Avx2Optimized,
        (GateUpLinearOptimizedDef<true>(DataType::BFloat(16), &GateUpLinearF32Bf16Avx2FmaEntry,
                                        &BuildGateUpLinearF32W16ReferenceArgs<true>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                                        "cpu::gate_up_linear_f32_wbf16_avx2_fma")))
#endif

#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuGateUpLinearF16Avx2Optimized,
        (GateUpLinearOptimizedDef<true>(DataType::Float(16), &GateUpLinearF32F16Avx2FmaEntry,
                                        &BuildGateUpLinearF32W16ReferenceArgs<false>, CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}),
                                        "cpu::gate_up_linear_f32_wf16_avx2_fma")))
#endif

} // namespace aethermind::cpu::detail
