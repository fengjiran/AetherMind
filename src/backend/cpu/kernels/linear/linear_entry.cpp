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

Status BuildLinearF32ReferenceArgs(const KernelParamsBuildContext& context,
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

    const auto row_count = ComputeRowCount(input, "LinearKernelEntry");
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

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            output, "LinearKernelEntry requires positive output strides"));
    if (rank > 2) {
        AM_RETURN_IF_ERROR(ValidateCollapsibleLeadingDimensions(output, "LinearKernelEntry"));
    }
    const int64_t output_row_stride = rank == 1 ? 0 : output.stride(rank - 2);
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "LinearKernelEntry", row_count.value(), out_features,
            output_row_stride, output.stride(rank - 1), "output"));
    AM_RETURN_IF_ERROR(ValidateNonOverlappingOutputRows(
            "LinearKernelEntry", row_count.value(), out_features,
            output_row_stride, output.stride(rank - 1)));

    if (in_features == 0) {
        ::new (params_buffer) LinearF32KernelArgs{
                .output = output.data<float>(),
                .row_count = row_count.value(),
                .in_features = 0,
                .out_features = out_features,
                .output_row_stride = output_row_stride,
                .output_col_stride = output.stride(rank - 1),
        };
        return Status::Ok();
    }

    if (input.data() == nullptr || weight.data() == nullptr || output.data() == nullptr) {
        return Status::InvalidArgument(
                "LinearKernelEntry requires non-null data pointers for non-empty tensors");
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            input, "LinearKernelEntry requires positive input strides"));
    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            weight, "LinearKernelEntry requires positive weight strides"));
    if (rank > 2) {
        AM_RETURN_IF_ERROR(ValidateCollapsibleLeadingDimensions(input, "LinearKernelEntry"));
    }

    const int64_t input_row_stride = rank == 1 ? 0 : input.stride(rank - 2);
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "LinearKernelEntry", row_count.value(), in_features,
            input_row_stride, input.stride(rank - 1), "input"));
    AM_RETURN_IF_ERROR(ValidateRowColMaxOffset(
            "LinearKernelEntry", out_features, in_features,
            weight.stride(0), weight.stride(1), "weight"));

    if (output.data() == input.data() || output.data() == weight.data()) {
        return Status::InvalidArgument("CPU Linear output must not alias input or weight");
    }

    ::new (params_buffer) LinearF32KernelArgs{
            .input = input.data<float>(),
            .weight = weight.data<float>(),
            .output = output.data<float>(),
            .row_count = row_count.value(),
            .in_features = in_features,
            .out_features = out_features,
            .input_row_stride = input_row_stride,
            .input_col_stride = input.stride(rank - 1),
            .weight_row_stride = weight.stride(0),
            .weight_col_stride = weight.stride(1),
            .output_row_stride = output_row_stride,
            .output_col_stride = output.stride(rank - 1),
    };
    return Status::Ok();
}

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
                .params_builder = &BuildLinearF32ReferenceArgs,
                .name = "cpu::linear_f32_reference"})

} // namespace aethermind::cpu::detail
