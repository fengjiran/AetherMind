#include "aethermind/backend/cpu/kernels/cpu_elementwise_mul_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/utils/overflow_check.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace aethermind {
namespace {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

auto GetParams(const void* kernel_params) noexcept {
    return static_cast<const CpuElementwiseMulParams*>(kernel_params);
}

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
                    std::string("CpuElementwiseMulKernel ") + name + " offset overflow");
        }
        int64_t new_max = 0;
        if (CheckOverflowAdd(max_offset, contrib, &new_max)) {
            return Status::InvalidArgument(
                    std::string("CpuElementwiseMulKernel ") + name + " offset overflow");
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
                    "CpuElementwiseMulKernel output element count overflow");
        }
        if (next < 0) {
            return Status::InvalidArgument(
                    "CpuElementwiseMulKernel output element count exceeds int64_t");
        }
        count = next;
    }
    return count;
}

Status ValidateAndExecute(const CpuElementwiseMulParams* params) noexcept {
    const TensorView& lhs = params->lhs_tensor;
    const TensorView& rhs = params->rhs_tensor;
    const MutableTensorView& output = params->output_tensor;

    if (!lhs.is_valid()) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires a valid lhs TensorView");
    }
    if (!rhs.is_valid()) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires a valid rhs TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires a valid output MutableTensorView");
    }

    if (lhs.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires float32 lhs TensorView");
    }
    if (rhs.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires float32 rhs TensorView");
    }
    if (output.dtype() != DataType::Make<float>()) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires float32 output MutableTensorView");
    }

    const int32_t output_rank = output.rank();
    const int32_t expected_rank = std::max(lhs.rank(), rhs.rank());
    if (output_rank != expected_rank) {
        return Status::InvalidArgument(
                "CpuElementwiseMulKernel output rank must equal max(lhs rank, rhs rank)");
    }
    if (output_rank > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument("CpuElementwiseMulKernel output rank exceeds maximum supported rank");
    }

    if (!ValidateBroadcastCompatible(lhs.shape(), rhs.shape(), output.shape())) {
        return Status::InvalidArgument(
                "CpuElementwiseMulKernel input shapes are not broadcast-compatible with output shape");
    }

    const auto numel_or = CheckedOutputNumel(output_rank, output.shape());
    if (!numel_or.ok()) return numel_or.status();
    const int64_t numel = numel_or.value();
    if (numel == 0) {
        return Status::Ok();
    }

    if (lhs.data() == nullptr) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires non-null lhs data");
    }
    if (rhs.data() == nullptr) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires non-null rhs data");
    }
    if (output.data() == nullptr) {
        return Status::InvalidArgument("CpuElementwiseMulKernel requires non-null output data");
    }

    {
        auto status = ValidateMaxOffset(lhs.rank(), lhs.shape(), lhs.strides(), "lhs");
        if (!status.ok()) return status;
        status = ValidateMaxOffset(rhs.rank(), rhs.shape(), rhs.strides(), "rhs");
        if (!status.ok()) return status;
        status = ValidateMaxOffset(output.rank(), output.shape(), output.strides(), "output");
        if (!status.ok()) return status;
    }

    const auto* const lhs_data = lhs.data<float>();
    const auto* const rhs_data = rhs.data<float>();
    auto* const output_data = output.data<float>();

    const auto lhs_shape = lhs.shape();
    const auto lhs_strides = lhs.strides();
    const auto rhs_shape = rhs.shape();
    const auto rhs_strides = rhs.strides();
    const auto out_shape = output.shape();
    const auto out_strides = output.strides();

    if (output_rank == 0) {
        output_data[0] = lhs_data[0] * rhs_data[0];
        return Status::Ok();
    }

    std::array<int64_t, kMaxRank> coord{};
    for (int64_t flat = 0; flat < numel; ++flat) {
        int64_t remaining = flat;
        for (int32_t axis = output_rank - 1; axis >= 0; --axis) {
            coord[axis] = remaining % out_shape[axis];
            remaining /= out_shape[axis];
        }

        const int64_t lhs_offset = MapCoordToOffset(lhs_shape, output_rank, lhs_strides, coord);
        const int64_t rhs_offset = MapCoordToOffset(rhs_shape, output_rank, rhs_strides, coord);

        int64_t out_offset = 0;
        for (int32_t axis = 0; axis < output_rank; ++axis) {
            out_offset += coord[axis] * out_strides[axis];
        }

        output_data[out_offset] = lhs_data[lhs_offset] * rhs_data[rhs_offset];
    }

    return Status::Ok();
}

}// namespace

Status CpuElementwiseMulKernel(const KernelContext& ctx) noexcept {
    const CpuElementwiseMulParams* params = GetParams(ctx.kernel_params);
    if (params == nullptr) {
        return Status::InvalidArgument(
                "CpuElementwiseMulKernel requires CpuElementwiseMulParams in KernelContext.kernel_params");
    }

    return ValidateAndExecute(params);
}

Status BuildCpuElementwiseMulParams(std::span<const TensorView> inputs,
                                        std::span<const MutableTensorView> outputs,
                                        void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("ElementwiseMul requires 2 inputs and 1 output");
    }
    ::new (params_buffer) CpuElementwiseMulParams{
            .lhs_tensor = inputs[0],
            .rhs_tensor = inputs[1],
            .output_tensor = outputs[0],
    };
    return Status::Ok();
}

AM_REGISTER_KERNEL(CpuElementwiseMulFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kElementwiseMul,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuElementwiseMulKernel,
                           .name = "cpu::elementwise_mul_f32_scalar",
                           .priority = 10,
                           .params_builder = &BuildCpuElementwiseMulParams,
                           .params_size = sizeof(CpuElementwiseMulParams),
                   })

}// namespace aethermind
