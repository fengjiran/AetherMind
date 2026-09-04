#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/shape_and_stride.h"
#include "elementwise_mul_internal.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>

namespace aethermind::cpu::detail {
namespace {

bool ValidateBroadcastCompatible(std::span<const int64_t> lhs_shape,
                                 std::span<const int64_t> rhs_shape,
                                 std::span<const int64_t> output_shape) noexcept {
    const auto output_rank = static_cast<int32_t>(output_shape.size());
    const auto lhs_offset = output_rank - static_cast<int32_t>(lhs_shape.size());
    const auto rhs_offset = output_rank - static_cast<int32_t>(rhs_shape.size());

    for (int32_t axis = 0; axis < output_rank; ++axis) {
        const int64_t out_dim = output_shape[axis];
        const int64_t lhs_dim = axis < lhs_offset ? 1 : lhs_shape[axis - lhs_offset];
        const int64_t rhs_dim = axis < rhs_offset ? 1 : rhs_shape[axis - rhs_offset];

        if (lhs_dim < 0 || rhs_dim < 0) return false;

        // Exact BroadcastShapes rule: lhs==1 → rhs; rhs==1 or equal → lhs.
        const int64_t expected = (lhs_dim == 1)                         ? rhs_dim
                                 : (rhs_dim == 1 || lhs_dim == rhs_dim) ? lhs_dim
                                                                        : int64_t{-1};
        if (expected < 0 || out_dim != expected) return false;
    }
    return true;
}

int64_t MapCoordToOffset(std::span<const int64_t> input_shape,
                         int32_t output_rank,
                         std::span<const int64_t> input_strides,
                         const std::array<int64_t, kMaxRank>& out_coord) noexcept {
    const auto input_rank = static_cast<int32_t>(input_shape.size());
    const int32_t axis_offset = output_rank - input_rank;
    int64_t offset = 0;
    for (int32_t axis = axis_offset; axis < output_rank; ++axis) {
        const int32_t input_axis = axis - axis_offset;
        const int64_t dim = input_shape[input_axis];
        const int64_t coord = (dim == 1) ? int64_t{0} : out_coord[axis];
        offset += coord * input_strides[input_axis];
    }
    return offset;
}

Status ValidateMaxOffset(int32_t rank,
                         std::span<const int64_t> shape,
                         std::span<const int64_t> strides,
                         const char* name) noexcept {
    if (rank == 0) return Status::Ok();

    int64_t max_offset = 0;
    for (int32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) return Status::Ok();
        int64_t contrib = 0;
        if (CheckOverflowMul(shape[i] - 1, strides[i], &contrib)) {
            return Status::InvalidArgument(
                    std::string("ElementwiseMulKernel ") + name + " offset overflow");
        }
        int64_t new_max = 0;
        if (CheckOverflowAdd(max_offset, contrib, &new_max)) {
            return Status::InvalidArgument(
                    std::string("ElementwiseMulKernel ") + name + " offset overflow");
        }
        max_offset = new_max;
    }
    return Status::Ok();
}

StatusOr<int64_t> CheckedOutputNumel(int32_t rank,
                                     std::span<const int64_t> shape) noexcept {
    if (rank == 0) return int64_t{1};

    int64_t count = 1;
    for (int32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return int64_t{0};
        }
        int64_t next = 0;
        if (CheckOverflowMul(count, shape[i], &next)) {
            return Status::InvalidArgument(
                    "ElementwiseMulKernel output element count overflow");
        }
        if (next < 0) {
            return Status::InvalidArgument(
                    "ElementwiseMulKernel output element count exceeds int64_t");
        }
        count = next;
    }
    return count;
}

Status ValidateAndBuildCommonElementwiseMulArgs(
        const KernelParamsBuildContext& context,
        ElementwiseMulKernelArgs& args) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument(
                "ElementwiseMulKernel requires 2 inputs and 1 output");
    }

    const TensorView& lhs = inputs[0];
    const TensorView& rhs = inputs[1];
    const MutableTensorView& output = outputs[0];

    if (!lhs.is_valid()) {
        return Status::InvalidArgument("ElementwiseMulKernel requires a valid lhs TensorView");
    }
    if (!rhs.is_valid()) {
        return Status::InvalidArgument("ElementwiseMulKernel requires a valid rhs TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument("ElementwiseMulKernel requires a valid output MutableTensorView");
    }

    if (lhs.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("ElementwiseMulKernel requires float32 lhs TensorView");
    }
    if (rhs.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("ElementwiseMulKernel requires float32 rhs TensorView");
    }
    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("ElementwiseMulKernel requires float32 output MutableTensorView");
    }

    const int32_t output_rank = output.rank();
    const int32_t expected_rank = std::max(lhs.rank(), rhs.rank());
    if (output_rank != expected_rank) {
        return Status::InvalidArgument(
                "ElementwiseMulKernel output rank must equal max(lhs rank, rhs rank)");
    }
    if (output_rank > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument("ElementwiseMulKernel output rank exceeds maximum supported rank");
    }

    if (!ValidateBroadcastCompatible(lhs.shape(), rhs.shape(), output.shape())) {
        return Status::InvalidArgument(
                "ElementwiseMulKernel input shapes are not broadcast-compatible with output shape");
    }

    const auto numel_or = CheckedOutputNumel(output_rank, output.shape());
    if (!numel_or.ok()) return numel_or.status();
    const int64_t numel = numel_or.value();
    if (numel == 0) {
        args = ElementwiseMulKernelArgs{};
        return Status::Ok();
    }

    if (lhs.data() == nullptr) {
        return Status::InvalidArgument("ElementwiseMulKernel requires non-null lhs data");
    }
    if (rhs.data() == nullptr) {
        return Status::InvalidArgument("ElementwiseMulKernel requires non-null rhs data");
    }
    if (output.data() == nullptr) {
        return Status::InvalidArgument("ElementwiseMulKernel requires non-null output data");
    }

    {
        auto status = ValidateMaxOffset(lhs.rank(), lhs.shape(), lhs.strides(), "lhs");
        if (!status.ok()) return status;
        status = ValidateMaxOffset(rhs.rank(), rhs.shape(), rhs.strides(), "rhs");
        if (!status.ok()) return status;
        status = ValidateMaxOffset(output.rank(), output.shape(), output.strides(), "output");
        if (!status.ok()) return status;
    }

    args.lhs_data = lhs.data<float>();
    args.rhs_data = rhs.data<float>();
    args.output_data = output.data<float>();
    args.numel = numel;
    args.lhs_rank = lhs.rank();
    args.rhs_rank = rhs.rank();
    args.output_rank = output_rank;

    for (int32_t i = 0; i < lhs.rank(); ++i) {
        args.lhs_shape[i] = lhs.shape()[i];
        args.lhs_strides[i] = lhs.strides()[i];
    }

    for (int32_t i = 0; i < rhs.rank(); ++i) {
        args.rhs_shape[i] = rhs.shape()[i];
        args.rhs_strides[i] = rhs.strides()[i];
    }

    for (int32_t i = 0; i < output_rank; ++i) {
        args.output_shape[i] = output.shape()[i];
        args.output_strides[i] = output.strides()[i];
    }

    return Status::Ok();
}

Status BuildElementwiseMulArgs(const KernelParamsBuildContext& context,
                               void* params_buffer) noexcept {
    ElementwiseMulKernelArgs args;
    AM_RETURN_IF_ERROR(ValidateAndBuildCommonElementwiseMulArgs(context, args));
    ::new (params_buffer) ElementwiseMulKernelArgs(args);
    return Status::Ok();
}

} // namespace

Status RunElementwiseMulReference(const ElementwiseMulKernelArgs& args) noexcept {
    if (args.numel == 0) {
        return Status::Ok();
    }

    if (args.output_rank == 0) {
        args.output_data[0] = args.lhs_data[0] * args.rhs_data[0];
        return Status::Ok();
    }

    std::array<int64_t, kMaxRank> coord{};
    for (int64_t flat = 0; flat < args.numel; ++flat) {
        int64_t remaining = flat;
        for (int32_t axis = args.output_rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % args.output_shape[axis];
            remaining /= args.output_shape[axis];
        }

        const int64_t lhs_offset = MapCoordToOffset(
                std::span<const int64_t>(args.lhs_shape.data(),
                                         static_cast<size_t>(args.lhs_rank)),
                args.output_rank,
                std::span<const int64_t>(args.lhs_strides.data(),
                                         static_cast<size_t>(args.lhs_rank)),
                coord);
        const int64_t rhs_offset = MapCoordToOffset(
                std::span<const int64_t>(args.rhs_shape.data(),
                                         static_cast<size_t>(args.rhs_rank)),
                args.output_rank,
                std::span<const int64_t>(args.rhs_strides.data(),
                                         static_cast<size_t>(args.rhs_rank)),
                coord);

        int64_t out_offset = 0;
        for (int32_t axis = 0; axis < args.output_rank; ++axis) {
            out_offset += coord[axis] * args.output_strides[axis];
        }

        args.output_data[out_offset] = args.lhs_data[lhs_offset] * args.rhs_data[rhs_offset];
    }

    return Status::Ok();
}

Status ElementwiseMulKernel(const KernelContext& ctx) noexcept {
    const auto* args = static_cast<const ElementwiseMulKernelArgs*>(
            ctx.kernel_params);
    AM_DCHECK(args != nullptr);
    return RunElementwiseMulReference(*args);
}

// The prepared args must satisfy the PreparedExecutionBindings params arena contract.
static_assert(std::is_trivially_destructible_v<ElementwiseMulKernelArgs>);
static_assert(alignof(ElementwiseMulKernelArgs) <= alignof(std::max_align_t));

AM_REGISTER_KERNEL(ElementwiseMulFp32Reference,
                   KernelDescriptor{
                           .op_type = OpType::kElementwiseMul,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &ElementwiseMulKernel,
                           .priority = 10,
                           .params_size = sizeof(ElementwiseMulKernelArgs),
                           .params_builder = &BuildElementwiseMulArgs,
                           .name = "cpu::elementwise_mul_f32_reference",
                   })

} // namespace aethermind::cpu::detail