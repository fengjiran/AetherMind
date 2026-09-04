#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "rmsnorm_internal.h"

#include <cmath>
#include <cstring>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

bool HasUnitColumnStrides(const RmsNormF32KernelArgs& args) noexcept {
    return args.input_col_stride == 1 && args.weight_stride == 1 && args.output_col_stride == 1;
}

bool HasIdenticalMapping(const TensorView& input,
                         const MutableTensorView& output) noexcept {
    if (input.data() != output.data() || input.dtype() != output.dtype() ||
        input.rank() != output.rank()) {
        return false;
    }

    for (int32_t i = 0; i < input.rank(); ++i) {
        if (input.dim(i) != output.dim(i) || input.stride(i) != output.stride(i)) {
            return false;
        }
    }

    return true;
}

template<typename KernelArgs>
Status ValidateAndBuildRmsNormArgs(const KernelParamsBuildContext& context,
                                   KernelArgs& args) noexcept {
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

    const StatusOr<int64_t> row_count = ComputeRowCount(input, "RmsNormKernelEntry");
    if (!row_count.ok()) {
        return row_count.status();
    }

    if (row_count.value() == 0) {
        args = KernelArgs{};
        args.row_count = 0;
        args.hidden_size = hidden_size;
        args.eps = eps;
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "RmsNormKernelEntry requires non-null data pointers for non-empty tensors");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            input, "RmsNormKernelEntry requires positive input strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            weight, "RmsNormKernelEntry requires positive weight strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            output, "RmsNormKernelEntry requires positive output strides"));

    if (rank > 2) {
        AM_RETURN_IF_ERROR(ValidateCollapsibleLeadingDimensions(input, "RmsNormKernelEntry"));
        AM_RETURN_IF_ERROR(ValidateCollapsibleLeadingDimensions(output, "RmsNormKernelEntry"));
    }

    const int64_t input_row_stride = rank == 1 ? 0 : input.stride(rank - 2);
    const int64_t output_row_stride = rank == 1 ? 0 : output.stride(rank - 2);
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "RmsNormKernelEntry",
            row_count.value(),
            hidden_size,
            input_row_stride,
            input.stride(rank - 1),
            "input"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "RmsNormKernelEntry",
            1,
            hidden_size,
            0,
            weight.stride(0),
            "weight"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "RmsNormKernelEntry",
            row_count.value(),
            hidden_size,
            output_row_stride,
            output.stride(rank - 1),
            "output"));

    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            "RmsNormKernelEntry",
            row_count.value(),
            hidden_size,
            output_row_stride,
            output.stride(rank - 1)));

    if (const bool shares_base_pointer = input.data() == output.data();
        shares_base_pointer && !HasIdenticalMapping(input, output)) {
        return Status::InvalidArgument(
                "CPU RmsNorm in-place execution requires identical "
                "input and output shape/strides");
    }

    if (weight.data() == output.data()) {
        return Status::InvalidArgument(
                "CPU RmsNorm output must not alias weight");
    }

    args.input = static_cast<decltype(args.input)>(input.data());
    args.weight = static_cast<decltype(args.weight)>(weight.data());
    args.output = static_cast<decltype(args.output)>(output.data());
    args.row_count = row_count.value();
    args.hidden_size = hidden_size;
    args.input_row_stride = input_row_stride;
    args.input_col_stride = input.stride(rank - 1);
    args.weight_stride = weight.stride(0);
    args.output_row_stride = output_row_stride;
    args.output_col_stride = output.stride(rank - 1);
    args.eps = eps;
    return Status::Ok();
}

Status ValidateAndBuildF32Args(const KernelParamsBuildContext& context,
                               RmsNormF32KernelArgs& args) noexcept {
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

    return ValidateAndBuildRmsNormArgs(context, args);
}

Status BuildRmsNormF32ReferenceArgs(const KernelParamsBuildContext& context,
                                    void* params_buffer) noexcept {
    RmsNormF32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildF32Args(context, args));
    ::new (params_buffer) RmsNormF32KernelArgs(args);
    return Status::Ok();
}

Status BuildRmsNormF32Avx2FmaArgs(const KernelParamsBuildContext& context,
                                  void* params_buffer) noexcept {
    RmsNormF32KernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildF32Args(context, args));

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
        KernelDescriptor{
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

#if defined(RMSNORM_HAS_AVX2_FMA_KERNEL)
AM_REGISTER_KERNEL(
        CpuRmsNormF32Avx2Fma,
        KernelDescriptor{
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
