#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "linear_internal.h"

#include <array>
#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

Status BuildLinearF32Args(const KernelParamsBuildContext& context,
                          void* params_buffer) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Linear requires 2 inputs and 1 output");
    }

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid()) {
        return Status::InvalidArgument("LinearKernelEntry requires a valid input TensorView");
    }
    if (!weight.is_valid()) {
        return Status::InvalidArgument("LinearKernelEntry requires a valid weight TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires a valid output MutableTensorView");
    }

    const DataType fp32 = DataType::Float32();
    if (input.dtype() != fp32 || weight.dtype() != fp32 || output.dtype() != fp32) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires float32 input, weight, and output TensorViews");
    }

    const int32_t rank = input.rank();
    if (rank < 1) {
        return Status::InvalidArgument("LinearKernelEntry requires input rank >= 1");
    }
    if (weight.rank() != 2) {
        return Status::InvalidArgument("LinearKernelEntry requires rank-2 weight TensorView");
    }
    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires output rank to match input rank");
    }

    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "LinearKernelEntry requires output leading dimensions to match input");
        }
    }

    const int64_t in_features = input.dim(rank - 1);
    const int64_t out_features = weight.dim(0);
    if (weight.dim(1) != in_features) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires weight input dimension to match input last dimension");
    }
    if (output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires output last dimension to match weight output dimension");
    }

    const auto row_count = ComputeFlattenedRowCount(input, "LinearKernelEntry");
    if (!row_count.ok()) {
        return row_count.status();
    }

    if (row_count.value() == 0 || out_features == 0) {
        ::new (params_buffer) LinearF32KernelArgs{
                .row_count = row_count.value(),
                .in_features = in_features,
                .out_features = out_features,
        };
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout("LinearKernelEntry", output_analysis));

    if (in_features == 0) {
        ::new (params_buffer) LinearF32KernelArgs{
                .output = output.data<float>(),
                .row_count = output_analysis.row_count(),
                .in_features = 0,
                .out_features = out_features,
                .output_row_stride = output_analysis.row_stride(),
                .output_col_stride = output_analysis.column_stride(),
        };
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires non-null data pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearKernelEntry input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis weight_analysis,
                        AnalyzeRowwiseView(weight, "LinearKernelEntry weight"));

    // Linear has no valid in-place form, not even when in_features equals
    // out_features: writing one output row can clobber input or weight elements
    // that later dot products have not read yet.
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU Linear", output_analysis.footprint(), "output", input_analysis.footprint(), "input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU Linear", output_analysis.footprint(), "output", weight_analysis.footprint(), "weight"));

    ::new (params_buffer) LinearF32KernelArgs{
            .input = input.data<float>(),
            .weight = weight.data<float>(),
            .output = output.data<float>(),
            .row_count = input_analysis.row_count(),
            .in_features = in_features,
            .out_features = out_features,
            .input_row_stride = input_analysis.row_stride(),
            .input_col_stride = input_analysis.column_stride(),
            .weight_row_stride = weight_analysis.row_stride(),
            .weight_col_stride = weight_analysis.column_stride(),
            .output_row_stride = output_analysis.row_stride(),
            .output_col_stride = output_analysis.column_stride(),
    };
    return Status::Ok();
}

StatusOr<LinearF32KernelArgs> BuildLinearF32PackedBase(
        const KernelParamsBuildContext& context,
        bool bpanel) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed Linear requires one activation, one output, and an artifact");
    }
    const TensorView& input = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!input.is_valid() || !output.is_valid()) {
        return Status::InvalidArgument(
                "Packed Linear requires valid input and output TensorViews");
    }
    if (input.dtype() != DataType::Float32() ||
        output.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "Packed Linear requires float32 input and output TensorViews");
    }
    if (input.rank() < 1 || packed.logical_shape.size() != 2U) {
        return Status::InvalidArgument(
                "Packed Linear requires rank >= 1 input and a rank-2 weight artifact");
    }

    const int64_t in_features = input.dim(input.rank() - 1);
    const int64_t out_features = packed.logical_shape[0];
    if (out_features < 0 || packed.logical_shape[1] != in_features) {
        return Status::InvalidArgument(
                "Packed Linear weight shape does not match the input feature dimension");
    }
    const std::array<int64_t, 2> expected_shape{out_features, in_features};
    if (bpanel) {
        AM_RETURN_IF_ERROR(ValidateBPanelF32PackedWeight(
                packed, expected_shape, "LinearKernelEntry"));
    } else {
        AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
                packed, expected_shape, "LinearKernelEntry"));
    }

    const int32_t rank = input.rank();
    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "Packed Linear requires output rank to match input rank");
    }
    for (int32_t dim = 0; dim < rank - 1; ++dim) {
        if (output.dim(dim) != input.dim(dim)) {
            return Status::InvalidArgument(
                    "Packed Linear requires output leading dimensions to match input");
        }
    }
    if (output.dim(rank - 1) != out_features) {
        return Status::InvalidArgument(
                "Packed Linear output feature dimension does not match the weight");
    }
    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "LinearKernelEntry"));

    LinearF32KernelArgs args{
            .row_count = row_count,
            .in_features = in_features,
            .out_features = out_features,
    };
    if (row_count == 0 || out_features == 0) {
        return args;
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "LinearKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "LinearKernelEntry output", output_analysis));
    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                "Packed Linear requires non-null output for non-empty tensors");
    }
    args.output = output.data<float>();
    args.output_row_stride = output_analysis.row_stride();
    args.output_col_stride = output_analysis.column_stride();

    if (in_features == 0) {
        return args;
    }
    if (input.data() == nullptr || packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed Linear requires non-null input and packed weight data");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "LinearKernelEntry input"));
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU Linear", output_analysis.footprint(), "output",
            input_analysis.footprint(), "input"));
    if (packed.nbytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow(
                "Packed Linear artifact byte range exceeds int64_t");
    }
    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                packed.data, static_cast<int64_t>(packed.nbytes),
                                size_t{1}, "LinearKernelEntry packed weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU Linear", output_analysis.footprint().envelope(), "output",
            packed_range, "packed weight"));

    args.input = input.data<float>();
    args.weight = static_cast<const float*>(packed.data);
    args.row_count = input_analysis.row_count();
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    args.weight_row_stride = in_features;
    args.weight_col_stride = 1;
    return args;
}

Status BuildLinearF32PackedIdentityArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const LinearF32KernelArgs args,
                        BuildLinearF32PackedBase(context, false));
    ::new (params_buffer) LinearF32KernelArgs(args);
    return Status::Ok();
}

Status BuildLinearF32BpanelArgs(const KernelParamsBuildContext& context,
                                void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const LinearF32KernelArgs base,
                        BuildLinearF32PackedBase(context, true));
    const int64_t n_blocks = base.out_features /
                                     cpu::kCpuBPanelF32V1NR +
                             (base.out_features % cpu::kCpuBPanelF32V1NR != 0);
    const PackedGemmF32Args args{
            .gemm = GemmF32Args{
                    .lhs = base.input,
                    .output = base.output,
                    .m = base.row_count,
                    .n = base.out_features,
                    .k = base.in_features,
                    .lhs_m_stride = base.input_row_stride,
                    .lhs_k_stride = base.input_col_stride,
                    .output_m_stride = base.output_row_stride,
                    .output_n_stride = base.output_col_stride,
            },
            .packed_b = base.weight,
            .packed_nbytes = context.packed_weight->nbytes,
            .logical_n = base.out_features,
            .logical_k = base.in_features,
            .weight_n_offset = 0,
            .n_blocks = n_blocks,
    };
    ::new (params_buffer) PackedGemmF32Args(args);
    return Status::Ok();
}

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status LinearF32PackedBEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const PackedGemmF32Args*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmF32PackedB(*args);
}
#endif

Status LinearF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunLinearF32Reference(*args);
}

} // namespace

static_assert(std::is_trivially_destructible_v<LinearF32KernelArgs>);
static_assert(alignof(LinearF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuLinearF32Reference,
        KernelDescriptor{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(LinearF32KernelArgs),
                .params_builder = &BuildLinearF32Args,
                .name = "cpu::linear_f32_reference"})


AM_REGISTER_KERNEL(
        CpuLinearF32PackedIdentityReference,
        KernelDescriptor{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &LinearF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(LinearF32KernelArgs),
                .params_builder = &BuildLinearF32PackedIdentityArgs,
                .name = "cpu::linear_f32_packed_identity_reference"})

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
static_assert(std::is_trivially_destructible_v<PackedGemmF32Args>);
static_assert(sizeof(PackedGemmF32Args) <= kMaxKernelParamsSize);
static_assert(alignof(PackedGemmF32Args) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuLinearF32PackedBpanelCandidate,
        KernelDescriptor{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = cpu::CpuBPanelF32V1Avx2Recipe(),
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &LinearF32PackedBEntry,
                .priority = 10,
                .params_size = sizeof(PackedGemmF32Args),
                .params_builder = &BuildLinearF32BpanelArgs,
                .name = "cpu::linear_f32_packed_bpanel_candidate"})
#endif

} // namespace aethermind::cpu::detail
