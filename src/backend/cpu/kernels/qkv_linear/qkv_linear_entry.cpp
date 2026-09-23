#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "qkv_linear_internal.h"
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

StatusOr<int64_t> TotalOutFeatures(const QkvLinearF32KernelMetadata& metadata) noexcept {
    if (metadata.q_out_features < 0 || metadata.k_out_features < 0 ||
        metadata.v_out_features < 0) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry requires non-negative output features");
    }

    int64_t qk_features = 0;
    int64_t total_features = 0;
    if (CheckOverflowAdd(metadata.q_out_features, metadata.k_out_features,
                         &qk_features) ||
        CheckOverflowAdd(qk_features, metadata.v_out_features,
                         &total_features)) {
        return Status::Overflow(
                "QkvLinearKernelEntry output feature count overflows int64_t");
    }
    return total_features;
}

StatusOr<QkvLinearF32KernelMetadata> ReadMetadata(
        std::span<const std::byte> attrs) noexcept {
    if (attrs.size() != sizeof(QkvLinearF32KernelMetadata)) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry requires QKV metadata in attrs");
    }

    QkvLinearF32KernelMetadata metadata{};
    std::memcpy(&metadata, attrs.data(), sizeof(metadata));
    AM_RETURN_IF_ERROR(TotalOutFeatures(metadata).status());
    return metadata;
}

Status BuildQkvLinearF32ReferenceArgs(const KernelParamsBuildContext& context,
                                      void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 3U) {
        return Status::InvalidArgument(
                "QkvLinear requires one activation input and three outputs");
    }
    if (!context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "QkvLinear requires a packed weight artifact");
    }

    AM_ASSIGN_OR_RETURN(const QkvLinearF32KernelMetadata metadata,
                        ReadMetadata(context.attrs));
    AM_ASSIGN_OR_RETURN(const int64_t total_out_features,
                        TotalOutFeatures(metadata));
    const TensorView& input = context.inputs[0];
    const MutableTensorView& query = context.outputs[0];
    const MutableTensorView& key = context.outputs[1];
    const MutableTensorView& value = context.outputs[2];

    if (!input.is_valid() || !query.is_valid() || !key.is_valid() || !value.is_valid()) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry requires valid TensorViews");
    }
    if (input.dtype() != DataType::Float32() || query.dtype() != DataType::Float32() ||
        key.dtype() != DataType::Float32() || value.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry requires float32 input and outputs");
    }

    const int32_t rank = input.rank();
    if (rank < 1 || query.rank() != rank || key.rank() != rank || value.rank() != rank) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry requires input and outputs with matching rank >= 1");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (query.dim(dim) != input.dim(dim) || key.dim(dim) != input.dim(dim) ||
            value.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "QkvLinearKernelEntry requires matching leading dimensions");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    if (query.dim(rank - 1) != metadata.q_out_features ||
        key.dim(rank - 1) != metadata.k_out_features ||
        value.dim(rank - 1) != metadata.v_out_features) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry outputs do not match QKV feature metadata");
    }
    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            *context.packed_weight, std::array{total_out_features, in_features},
            "QkvLinearKernelEntry"));

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "QkvLinearKernelEntry"));
    if (row_count == 0) {
        ::new (params_buffer) QkvLinearF32KernelArgs{
                .row_count = 0,
                .in_features = in_features,
                .q_out_features = metadata.q_out_features,
                .k_out_features = metadata.k_out_features,
                .v_out_features = metadata.v_out_features,
        };
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis query_analysis,
                        AnalyzeRowwiseView(query, "QkvLinearKernelEntry q output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis key_analysis,
                        AnalyzeRowwiseView(key, "QkvLinearKernelEntry k output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis value_analysis,
                        AnalyzeRowwiseView(value, "QkvLinearKernelEntry v output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "QkvLinearKernelEntry q output", query_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "QkvLinearKernelEntry k output", key_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "QkvLinearKernelEntry v output", value_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", query_analysis.footprint(), "q output",
            key_analysis.footprint(), "k output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", query_analysis.footprint(), "q output",
            value_analysis.footprint(), "v output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", key_analysis.footprint(), "k output",
            value_analysis.footprint(), "v output"));

    QkvLinearF32KernelArgs args{
            .query = query.data<float>(),
            .key = key.data<float>(),
            .value = value.data<float>(),
            .row_count = row_count,
            .in_features = in_features,
            .q_out_features = metadata.q_out_features,
            .k_out_features = metadata.k_out_features,
            .v_out_features = metadata.v_out_features,
            .query_row_stride = query_analysis.row_stride(),
            .query_col_stride = query_analysis.column_stride(),
            .key_row_stride = key_analysis.row_stride(),
            .key_col_stride = key_analysis.column_stride(),
            .value_row_stride = value_analysis.row_stride(),
            .value_col_stride = value_analysis.column_stride(),
    };

    if (in_features == 0 || total_out_features == 0) {
        ::new (params_buffer) QkvLinearF32KernelArgs(args);
        return Status::Ok();
    }

    if (input.data() == nullptr ||
        (metadata.q_out_features != 0 && query.data() == nullptr) ||
        (metadata.k_out_features != 0 && key.data() == nullptr) ||
        (metadata.v_out_features != 0 && value.data() == nullptr) ||
        context.packed_weight->data == nullptr) {
        return Status::InvalidArgument(
                "QkvLinearKernelEntry requires non-null data for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "QkvLinearKernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", query_analysis.footprint(), "q output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", key_analysis.footprint(), "k output",
            input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", value_analysis.footprint(), "v output",
            input_analysis.footprint(), "input"));

    if (context.packed_weight->nbytes >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow(
                "QkvLinearKernelEntry packed storage size exceeds int64_t");
    }
    AM_ASSIGN_OR_RETURN(
            const ByteAddressRange packed_range,
            BuildContiguousByteRange(context.packed_weight->data,
                                     static_cast<int64_t>(context.packed_weight->nbytes),
                                     size_t{1}, "QkvLinearKernelEntry packed weight"));
    for (const auto& [footprint, role]: std::array{
                 std::pair{query_analysis.footprint(), "q output"},
                 std::pair{key_analysis.footprint(), "k output"},
                 std::pair{value_analysis.footprint(), "v output"}}) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU QkvLinear", footprint, role, packed_range,
                "packed weight"));
    }

    int64_t key_weight_offset = 0;
    int64_t value_weight_row = 0;
    int64_t value_weight_offset = 0;
    if (CheckOverflowMul(metadata.q_out_features, in_features, &key_weight_offset) ||
        CheckOverflowAdd(metadata.q_out_features, metadata.k_out_features,
                         &value_weight_row) ||
        CheckOverflowMul(value_weight_row, in_features, &value_weight_offset)) {
        return Status::Overflow(
                "QkvLinearKernelEntry packed weight offsets overflow int64_t");
    }

    args.input = input.data<float>();
    const auto* packed_data = static_cast<const float*>(context.packed_weight->data);
    args.q_weight = packed_data;
    args.k_weight = packed_data + key_weight_offset;
    args.v_weight = packed_data + value_weight_offset;
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    ::new (params_buffer) QkvLinearF32KernelArgs(args);
    return Status::Ok();
}

Status BuildQkvLinearF32BpanelArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 3U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed QkvLinear requires one activation, three outputs, and an artifact");
    }
    AM_ASSIGN_OR_RETURN(const QkvLinearF32KernelMetadata metadata,
                        ReadMetadata(context.attrs));
    AM_ASSIGN_OR_RETURN(const int64_t total_n, TotalOutFeatures(metadata));
    const TensorView& input = context.inputs[0];
    const MutableTensorView& query = context.outputs[0];
    const MutableTensorView& key = context.outputs[1];
    const MutableTensorView& value = context.outputs[2];
    if (!input.is_valid() || !query.is_valid() || !key.is_valid() ||
        !value.is_valid() || input.dtype() != DataType::Float32() ||
        query.dtype() != DataType::Float32() || key.dtype() != DataType::Float32() ||
        value.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "Packed QkvLinear requires valid float32 activation and outputs");
    }
    const int32_t rank = input.rank();
    if (rank < 1 || query.rank() != rank || key.rank() != rank || value.rank() != rank) {
        return Status::InvalidArgument(
                "Packed QkvLinear requires matching input/output rank >= 1");
    }
    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (query.dim(dim) != input.dim(dim) || key.dim(dim) != input.dim(dim) ||
            value.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "Packed QkvLinear requires matching leading dimensions");
        }
    }
    const int64_t in_features = input.dim(rank - 1);
    if (query.dim(rank - 1) != metadata.q_out_features ||
        key.dim(rank - 1) != metadata.k_out_features ||
        value.dim(rank - 1) != metadata.v_out_features) {
        return Status::InvalidArgument(
                "Packed QkvLinear outputs do not match metadata");
    }
    AM_RETURN_IF_ERROR(ValidateBPanelF32PackedWeight(
            *context.packed_weight, std::array{total_n, in_features},
            "QkvLinearKernelEntry"));
    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "QkvLinearKernelEntry"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis q_analysis,
                        AnalyzeRowwiseView(query, "QkvLinearKernelEntry q output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis k_analysis,
                        AnalyzeRowwiseView(key, "QkvLinearKernelEntry k output"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis v_analysis,
                        AnalyzeRowwiseView(value, "QkvLinearKernelEntry v output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "QkvLinearKernelEntry q output", q_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "QkvLinearKernelEntry k output", k_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "QkvLinearKernelEntry v output", v_analysis));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", q_analysis.footprint(), "q output",
            k_analysis.footprint(), "k output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", q_analysis.footprint(), "q output",
            v_analysis.footprint(), "v output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU QkvLinear", k_analysis.footprint(), "k output",
            v_analysis.footprint(), "v output"));

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
    QkvLinearF32PackedBKernelArgs args{
            .query = build_gemm(query, q_analysis, metadata.q_out_features, 0),
            .key = build_gemm(key, k_analysis, metadata.k_out_features,
                              metadata.q_out_features),
            .value = build_gemm(
                    value, v_analysis, metadata.v_out_features,
                    metadata.q_out_features + metadata.k_out_features),
    };
    if (row_count == 0 || total_n == 0 || in_features == 0) {
        ::new (params_buffer) QkvLinearF32PackedBKernelArgs(args);
        return Status::Ok();
    }
    if (input.data() == nullptr ||
        (metadata.q_out_features != 0 && query.data() == nullptr) ||
        (metadata.k_out_features != 0 && key.data() == nullptr) ||
        (metadata.v_out_features != 0 && value.data() == nullptr)) {
        return Status::InvalidArgument(
                "Packed QkvLinear requires non-null data for non-empty tensors");
    }
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "QkvLinearKernelEntry input"));
    for (const auto& [footprint, role]: std::array{
                 std::pair{q_analysis.footprint(), "q output"},
                 std::pair{k_analysis.footprint(), "k output"},
                 std::pair{v_analysis.footprint(), "v output"}}) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
                "CPU QkvLinear", footprint, role,
                input_analysis.footprint(), "input"));
    }
    if (context.packed_weight->nbytes >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow(
                "QkvLinearKernelEntry packed storage size exceeds int64_t");
    }
    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                context.packed_weight->data,
                                static_cast<int64_t>(context.packed_weight->nbytes),
                                size_t{1}, "QkvLinearKernelEntry packed weight"));
    for (const auto& [footprint, role]: std::array{
                 std::pair{q_analysis.footprint(), "q output"},
                 std::pair{k_analysis.footprint(), "k output"},
                 std::pair{v_analysis.footprint(), "v output"}}) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjointFromContiguous(
                "CPU QkvLinear", footprint, role, packed_range,
                "packed weight"));
    }
    if (row_count > 0) {
        args.query.gemm.lhs_m_stride = input_analysis.row_stride();
        args.key.gemm.lhs_m_stride = input_analysis.row_stride();
        args.value.gemm.lhs_m_stride = input_analysis.row_stride();
        args.query.gemm.lhs_k_stride = input_analysis.column_stride();
        args.key.gemm.lhs_k_stride = input_analysis.column_stride();
        args.value.gemm.lhs_k_stride = input_analysis.column_stride();
    }
    ::new (params_buffer) QkvLinearF32PackedBKernelArgs(args);
    return Status::Ok();
}

Status QkvLinearF32PackedBEntry(const KernelContext& context) noexcept {
    const auto* args =
            static_cast<const QkvLinearF32PackedBKernelArgs*>(context.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunQkvLinearF32PackedB(*args);
}

Status BuildQkvLinearF32Metadata(const OpParams& params,
                                 std::vector<std::byte>& attrs) {
    const auto* qkv_params = std::get_if<QkvLinearParams>(&params);
    if (qkv_params == nullptr || qkv_params->has_bias) {
        return Status::InvalidArgument(
                "CPU QkvLinear kernel requires no-bias QkvLinearParams");
    }

    const QkvLinearF32KernelMetadata metadata{
            .q_out_features = qkv_params->q_out_features,
            .k_out_features = qkv_params->k_out_features,
            .v_out_features = qkv_params->v_out_features,
    };

    AM_RETURN_IF_ERROR(TotalOutFeatures(metadata).status());
    const auto bytes = std::as_bytes(std::span{&metadata, size_t{1}});
    attrs.assign(bytes.begin(), bytes.end());
    return Status::Ok();
}

Status QkvLinearF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const QkvLinearF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunQkvLinearF32Reference(*args);
}

} // namespace

Status RunQkvLinearF32PackedB(
        const QkvLinearF32PackedBKernelArgs& args) noexcept {
    AM_RETURN_IF_ERROR(RunGemmF32PackedB(args.query));
    AM_RETURN_IF_ERROR(RunGemmF32PackedB(args.key));
    return RunGemmF32PackedB(args.value);
}

static_assert(std::is_trivially_copyable_v<QkvLinearF32KernelMetadata>);
static_assert(std::is_trivially_destructible_v<QkvLinearF32KernelArgs>);
static_assert(sizeof(QkvLinearF32KernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(QkvLinearF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuQkvLinearF32Reference,
        KernelDescriptor{
                .op_type = OpType::kQkvLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &QkvLinearF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(QkvLinearF32KernelArgs),
                .params_builder = &BuildQkvLinearF32ReferenceArgs,
                .metadata_builder = &BuildQkvLinearF32Metadata,
                .name = "cpu::qkv_linear_f32_reference"})


#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
static_assert(std::is_trivially_destructible_v<QkvLinearF32PackedBKernelArgs>);
static_assert(sizeof(QkvLinearF32PackedBKernelArgs) <= kMaxKernelParamsSize);
static_assert(alignof(QkvLinearF32PackedBKernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuQkvLinearF32PackedBpanelCandidate,
        KernelDescriptor{
                .op_type = OpType::kQkvLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = cpu::CpuBPanelF32V1Avx2Recipe(),
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &QkvLinearF32PackedBEntry,
                .priority = 10,
                .params_size = sizeof(QkvLinearF32PackedBKernelArgs),
                .params_builder = &BuildQkvLinearF32BpanelArgs,
                .metadata_builder = &BuildQkvLinearF32Metadata,
                .name = "cpu::qkv_linear_f32_packed_bpanel_candidate"})
#endif

} // namespace aethermind::cpu::detail
