#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "rmsnorm_internal.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

bool HasUnitColumnStrides(const RmsNormF32KernelArgs& args) noexcept {
    return args.input_col_stride == 1 && args.weight_stride == 1 && args.output_col_stride == 1;
}

template<typename KernelArgs>
StatusOr<KernelArgs> ValidateAndBuildRmsNormArgs(
        const KernelParamsBuildContext& context) noexcept {
    KernelArgs args{};
    float eps = 0.0f;
    if (context.attrs.size() != sizeof(float)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }

    std::memcpy(&eps, context.attrs.data(), sizeof(float));
    if (!std::isfinite(eps) || eps <= 0.0F) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires finite positive epsilon");
    }

    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("RmsNorm requires 2 inputs and 1 output");
    }

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const MutableTensorView& output = outputs[0];
    if (!input.is_valid()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires a valid input TensorView");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires a valid weight TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires a valid output MutableTensorView");
    }

    const int32_t rank = input.rank();
    if (rank < 1) {
        return Status::InvalidArgument("RmsNormKernelEntry requires input rank >= 1");
    }

    if (weight.rank() != 1) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires rank-1 weight TensorView");
    }

    if (output.rank() != rank) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires output rank to match input rank");
    }

    for (int32_t i = 0; i < rank; ++i) {
        if (output.dim(i) != input.dim(i)) {
            return Status::InvalidArgument(
                    "RmsNormKernelEntry requires output shape to match input shape");
        }
    }

    const int64_t hidden_size = input.dim(rank - 1);
    if (hidden_size <= 0) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires positive hidden_size");
    }

    if (weight.dim(0) != hidden_size) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires weight length to match hidden_size");
    }

    const StatusOr<int64_t> row_count = ComputeFlattenedRowCount(input, "RmsNormKernelEntry");
    if (!row_count.ok()) {
        return row_count.status();
    }

    if (row_count.value() == 0) {
        args.row_count = 0;
        args.hidden_size = hidden_size;
        args.eps = eps;
        return args;
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires non-null data pointers for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "RmsNormKernelEntry input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis weight_analysis,
                        AnalyzeRowwiseView(weight, "RmsNormKernelEntry weight"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "RmsNormKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout("RmsNormKernelEntry", output_analysis));

    // Exact in-place is the only input aliasing RmsNorm can execute correctly:
    // each output element is written after its own input element was read.
    if (!HaveIdenticalViewMapping(input, output)) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
                "CPU RmsNorm", output_analysis.footprint(), "output", input_analysis.footprint(), "input"));
    }

    // The weight is re-read for every row, so no output element may land in it.
    AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
            "CPU RmsNorm", output_analysis.footprint(), "output", weight_analysis.footprint(), "weight"));

    args.input = static_cast<decltype(args.input)>(input.data());
    args.weight = static_cast<decltype(args.weight)>(weight.data());
    args.output = static_cast<decltype(args.output)>(output.data());
    args.row_count = input_analysis.row_count();
    args.hidden_size = hidden_size;
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    args.weight_stride = weight_analysis.column_stride();
    args.output_row_stride = output_analysis.row_stride();
    args.output_col_stride = output_analysis.column_stride();
    args.eps = eps;
    return args;
}

StatusOr<RmsNormF32KernelArgs> ValidateAndBuildF32Args(
        const KernelParamsBuildContext& context) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("RmsNorm requires 2 inputs and 1 output");
    }

    if (inputs[0].dtype() != DataType::Float32() || inputs[1].dtype() != DataType::Float32() ||
        outputs[0].dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires float32 input, weight, and output TensorViews");
    }

    return ValidateAndBuildRmsNormArgs<RmsNormF32KernelArgs>(context);
}

Status BuildRmsNormF32ReferenceArgs(const KernelParamsBuildContext& context,
                                    void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const RmsNormF32KernelArgs args, ValidateAndBuildF32Args(context));
    ::new (params_buffer) RmsNormF32KernelArgs(args);
    return Status::Ok();
}

Status BuildRmsNormF32PackedIdentityArgs(
        const KernelParamsBuildContext& context,
        void* params_buffer) noexcept {
    if (context.inputs.size() != 1U || context.outputs.size() != 1U ||
        !context.packed_weight.has_value()) {
        return Status::InvalidArgument(
                "Packed RmsNorm requires one input, one output, and a weight artifact");
    }
    if (context.attrs.size() != sizeof(float)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires epsilon in KernelContext.attrs");
    }
    float eps = 0.0F;
    std::memcpy(&eps, context.attrs.data(), sizeof(eps));
    if (!std::isfinite(eps) || eps <= 0.0F) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires finite positive epsilon");
    }

    const TensorView& input = context.inputs[0];
    const MutableTensorView& output = context.outputs[0];
    const PackedWeightView& packed = *context.packed_weight;
    if (!input.is_valid() || !output.is_valid() ||
        input.dtype() != DataType::Float32() ||
        output.dtype() != DataType::Float32() || input.rank() < 1 ||
        output.rank() != input.rank() || packed.logical_shape.size() != 1U) {
        return Status::InvalidArgument(
                "Packed RmsNorm requires rank-matched float32 activation views and rank-1 weight");
    }
    for (int32_t i = 0; i < input.rank(); ++i) {
        if (output.dim(i) != input.dim(i)) {
            return Status::InvalidArgument(
                    "Packed RmsNorm output shape must match input shape");
        }
    }
    const int64_t hidden_size = input.dim(input.rank() - 1);
    if (hidden_size <= 0 || packed.logical_shape[0] != hidden_size) {
        return Status::InvalidArgument(
                "Packed RmsNorm weight length must match positive hidden size");
    }
    const std::array<int64_t, 1> expected_shape{hidden_size};
    AM_RETURN_IF_ERROR(ValidateIdentityPackedWeight(
            packed, expected_shape, "RmsNormKernelEntry"));

    AM_ASSIGN_OR_RETURN(const int64_t row_count,
                        ComputeFlattenedRowCount(input, "RmsNormKernelEntry"));
    RmsNormF32KernelArgs args{
            .row_count = row_count,
            .hidden_size = hidden_size,
            .weight_stride = 1,
            .eps = eps,
    };
    if (row_count == 0) {
        ::new (params_buffer) RmsNormF32KernelArgs(args);
        return Status::Ok();
    }
    if (input.data() == nullptr || output.data() == nullptr || packed.data == nullptr) {
        return Status::InvalidArgument(
                "Packed RmsNorm requires non-null storage for non-empty tensors");
    }

    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis input_analysis,
                        AnalyzeRowwiseView(input, "RmsNormKernelEntry input"));
    AM_ASSIGN_OR_RETURN(const RowwiseViewAnalysis output_analysis,
                        AnalyzeRowwiseView(output, "RmsNormKernelEntry output"));
    AM_RETURN_IF_ERROR(ValidateRowwiseOutputLayout(
            "RmsNormKernelEntry", output_analysis));
    if (!HaveIdenticalViewMapping(input, output)) {
        AM_RETURN_IF_ERROR(ValidateRowwiseDisjoint(
                "CPU RmsNorm", output_analysis.footprint(), "output",
                input_analysis.footprint(), "input"));
    }
    if (packed.nbytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow("Packed RmsNorm artifact byte range exceeds int64_t");
    }
    AM_ASSIGN_OR_RETURN(const ByteAddressRange packed_range,
                        BuildContiguousByteRange(
                                packed.data, static_cast<int64_t>(packed.nbytes),
                                size_t{1}, "RmsNormKernelEntry packed weight"));
    AM_RETURN_IF_ERROR(ValidateContiguousDisjoint(
            "CPU RmsNorm", output_analysis.footprint().envelope(), "output",
            packed_range, "packed weight"));

    args.input = input.data<float>();
    args.weight = static_cast<const float*>(packed.data);
    args.output = output.data<float>();
    args.row_count = input_analysis.row_count();
    args.input_row_stride = input_analysis.row_stride();
    args.input_col_stride = input_analysis.column_stride();
    args.output_row_stride = output_analysis.row_stride();
    args.output_col_stride = output_analysis.column_stride();
    ::new (params_buffer) RmsNormF32KernelArgs(args);
    return Status::Ok();
}

Status BuildRmsNormF32Avx2FmaArgs(const KernelParamsBuildContext& context,
                                  void* params_buffer) noexcept {
    AM_ASSIGN_OR_RETURN(const RmsNormF32KernelArgs args, ValidateAndBuildF32Args(context));

    if (args.row_count != 0 && !HasUnitColumnStrides(args)) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry AVX2 requires unit column strides");
    }

    ::new (params_buffer) RmsNormF32KernelArgs(args);
    return Status::Ok();
}

Status BuildRmsNormF32Metadata(const OpParams& params,
                               std::vector<std::byte>& attrs) {
    const auto* rmsnorm_params = std::get_if<RmsNormParams>(&params);
    if (rmsnorm_params == nullptr) {
        return Status::InvalidArgument("RmsNorm kernel requires RmsNormParams");
    }

    if (!std::isfinite(rmsnorm_params->eps) || rmsnorm_params->eps <= 0.0F) {
        return Status::InvalidArgument("RmsNorm kernel requires finite positive epsilon");
    }

    const auto eps_bytes = std::as_bytes(std::span{&rmsnorm_params->eps, size_t{1}});
    attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status RmsNormF32ReferenceEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const RmsNormF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunRmsNormF32Reference(*args);
}

#if defined(RMSNORM_HAS_AVX2_FMA_KERNEL)
Status RmsNormF32Avx2FmaEntry(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const RmsNormF32KernelArgs*>(ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunRmsNormF32Avx2Fma(*args);
}
#endif

} // namespace

static_assert(std::is_trivially_destructible_v<RmsNormF32KernelArgs>);
static_assert(alignof(RmsNormF32KernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(
        CpuRmsNormF32Reference,
        KernelDef{
                .op_type = OpType::kRmsNorm,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .kernel_func = &RmsNormF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(RmsNormF32KernelArgs),
                .params_builder = &BuildRmsNormF32ReferenceArgs,
                .metadata_builder = &BuildRmsNormF32Metadata,
                .name = "cpu::rmsnorm_f32_reference"});

AM_REGISTER_KERNEL(
        CpuRmsNormF32PackedIdentityReference,
        KernelDef{
                .op_type = OpType::kRmsNorm,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPacked,
                        .phase = ExecPhase::kBoth,
                },
                .packing_recipe = CpuIdentityPackingRecipe(),
                .kernel_func = &RmsNormF32ReferenceEntry,
                .priority = 10,
                .params_size = sizeof(RmsNormF32KernelArgs),
                .params_builder = &BuildRmsNormF32PackedIdentityArgs,
                .metadata_builder = &BuildRmsNormF32Metadata,
                .name = "cpu::rmsnorm_f32_packed_identity_reference"});

#if defined(RMSNORM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuRmsNormF32Avx2Fma,
        KernelDef{
                .op_type = OpType::kRmsNorm,
                .selector = KernelSelector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .weight_dtype = DataType::Float32(),
                        .weight_format = WeightFormat::kPlain,
                        .phase = ExecPhase::kBoth,
                },
                .cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
                .kernel_func = &RmsNormF32Avx2FmaEntry,
                .priority = 20,
                .params_size = sizeof(RmsNormF32KernelArgs),
                .params_builder = &BuildRmsNormF32Avx2FmaArgs,
                .metadata_builder = &BuildRmsNormF32Metadata,
                .name = "cpu::rmsnorm_f32_avx2_fma"});
#endif

} // namespace aethermind::cpu::detail
