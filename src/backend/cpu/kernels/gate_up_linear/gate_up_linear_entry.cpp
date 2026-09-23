#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "gate_up_linear_internal.h"
#include "utils/overflow_check.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
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

Status BuildGateUpLinearF32ReferenceArgs(const KernelParamsBuildContext& context,
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
        ::new (params_buffer) GateUpLinearF32KernelArgs{
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

    GateUpLinearF32KernelArgs args{
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
        ::new (params_buffer) GateUpLinearF32KernelArgs(args);
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
        ::new (params_buffer) GateUpLinearF32KernelArgs(args);
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

    if (context.packed_weight->nbytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
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
    ::new (params_buffer) GateUpLinearF32KernelArgs(args);
    return Status::Ok();
}

Status BuildGateUpLinearF32BpanelArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 2U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed GateUpLinear requires one activation, two outputs, and an artifact");
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

    const int64_t n_blocks = total_n / cpu::kCpuBPanelF32V1NR +
                             (total_n % cpu::kCpuBPanelF32V1NR != 0);
    const float* const packed_data =
            static_cast<const float*>(context.packed_weight->data);
    const auto build_gemm = [&](const MutableTensorView& output,
                                const RowwiseViewAnalysis& output_analysis,
                                int64_t output_n,
                                int64_t n_offset) {
        return PackedGemmF32Args{
                .gemm = GemmF32Args{
                        .lhs = input.data<float>(),
                        .output = output.data<float>(),
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
    GateUpLinearF32PackedBKernelArgs args{
            .gate = build_gemm(gate, gate_analysis,
                               metadata.gate_out_features, 0),
            .up = build_gemm(up, up_analysis,
                             metadata.up_out_features,
                             metadata.gate_out_features),
    };
    if (row_count == 0 || total_n == 0 || in_features == 0) {
        ::new (params_buffer) GateUpLinearF32PackedBKernelArgs(args);
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
    if (context.packed_weight->nbytes >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
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
    ::new (params_buffer) GateUpLinearF32PackedBKernelArgs(args);
    return Status::Ok();
}

Status GateUpLinearF32PackedBEntry(const KernelContext& context) noexcept {
    const auto* args = static_cast<const GateUpLinearF32PackedBKernelArgs*>(
            context.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32PackedB(*args);
}

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

Status GateUpLinearF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const GateUpLinearF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGateUpLinearF32Reference(*args);
}

} // namespace

Status RunGateUpLinearF32PackedB(
        const GateUpLinearF32PackedBKernelArgs& args) noexcept {
    AM_RETURN_IF_ERROR(RunGemmF32PackedB(args.gate));
    return RunGemmF32PackedB(args.up);
}

static_assert(std::is_trivially_copyable_v<GateUpLinearF32KernelMetadata>);
static_assert(std::is_trivially_destructible_v<GateUpLinearF32KernelArgs>);
static_assert(sizeof(GateUpLinearF32KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(GateUpLinearF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuGateUpLinearF32Reference,
        KernelDescriptor{
                .op_type = OpType::kGateUpLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &GateUpLinearF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(GateUpLinearF32KernelArgs),
                .params_builder = &BuildGateUpLinearF32ReferenceArgs,
                .metadata_builder = &BuildGateUpLinearF32Metadata,
                .name = "cpu::gate_up_linear_f32_reference"})


#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
static_assert(std::is_trivially_destructible_v<GateUpLinearF32PackedBKernelArgs>);
static_assert(sizeof(GateUpLinearF32PackedBKernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(GateUpLinearF32PackedBKernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuGateUpLinearF32PackedBpanelCandidate,
        KernelDescriptor{
                .op_type = OpType::kGateUpLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = cpu::CpuBPanelF32V1Avx2Recipe(),
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &GateUpLinearF32PackedBEntry,
                .priority = 10,
                .params_size = sizeof(GateUpLinearF32PackedBKernelArgs),
                .params_builder = &BuildGateUpLinearF32BpanelArgs,
                .metadata_builder = &BuildGateUpLinearF32Metadata,
                .name = "cpu::gate_up_linear_f32_packed_bpanel_candidate"})
#endif

} // namespace aethermind::cpu::detail
