#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "linear_internal.h"

#include <cstddef>
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

Status LinearF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunLinearF32Reference(*args);
}

#if defined(AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE) || defined(GEMM_HAS_AVX2_FMA_KERNEL)
GemmF32Args MakeLinearF32GemmArgs(const LinearF32KernelArgs& args) noexcept {
    return GemmF32Args{
            .lhs = args.input,
            .rhs = args.weight,
            .output = args.output,
            .m = args.row_count,
            .n = args.out_features,
            .k = args.in_features,
            .lhs_m_stride = args.input_row_stride,
            .lhs_k_stride = args.input_col_stride,
            .rhs_k_stride = args.weight_col_stride,
            .rhs_n_stride = args.weight_row_stride,
            .output_m_stride = args.output_row_stride,
            .output_n_stride = args.output_col_stride,
    };
}
#endif

#if defined(AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE)
Status LinearF32ScalarCandidateEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmF32ScalarOptimized(MakeLinearF32GemmArgs(*args));
}
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status LinearF32Avx2FmaCandidateEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const LinearF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunGemmF32Avx2Fma(MakeLinearF32GemmArgs(*args));
}
#endif

} // namespace

static_assert(std::is_trivially_destructible_v<LinearF32KernelArgs>);
static_assert(alignof(LinearF32KernelArgs) <= alignof(std::max_align_t));

#if defined(AETHERMIND_ENABLE_GEMM_SCALAR_CANDIDATE)
AM_REGISTER_KERNEL(
        CpuLinearF32ScalarCandidate,
        KernelDescriptor{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &LinearF32ScalarCandidateEntry,
                .priority = 10,
                .params_size = sizeof(LinearF32KernelArgs),
                .params_builder = &BuildLinearF32Args,
                .name = "cpu::linear_f32_scalar_candidate"})
#else
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
#endif

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuLinearF32Avx2FmaCandidate,
        KernelDescriptor{
                .op_type = OpType::kLinear,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &LinearF32Avx2FmaCandidateEntry,
                .priority = 20,
                .params_size = sizeof(LinearF32KernelArgs),
                .params_builder = &BuildLinearF32Args,
                .name = "cpu::linear_f32_avx2_fma_candidate"})
#endif

} // namespace aethermind::cpu::detail
