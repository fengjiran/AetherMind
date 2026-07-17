// Kernel entry for the CPU Add operator (directory-structured).
//
// Validates CpuAddParams (dtypes, shapes, broadcast compatibility, numel,
// pointers, max offsets) without dynamic allocation, then dispatches to
// cpu::detail::AddKernel_CPU_Scalar for execution across all five supported
// same-dtype configurations. Kernel registration with five canonical
// selectors (weight_dtype == act_dtype) lives here via AM_REGISTER_KERNEL.

#include "add_internal.h"

#include "aethermind/backend/cpu/kernels/add/cpu_add_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/utils/overflow_check.h"

#include <cstdint>
#include <span>
#include <string>

namespace aethermind {
namespace {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

bool IsSupportedAddDType(const DataType& dtype) noexcept {
    return dtype == DataType::Float32() ||
           dtype == DataType::Double() ||
           dtype == DataType::BFloat(16) ||
           dtype == DataType::Int(32) ||
           dtype == DataType::Int(64);
}

auto GetParams(const void* kernel_params) noexcept {
    return static_cast<const CpuAddParams*>(kernel_params);
}

// Inline broadcast-compatibility check:
// for each axis, lhs==1 → rhs dim; rhs==1 or equal → lhs dim.
// Any mismatch returns false.
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

        // Broadcast rule: lhs==1 → rhs; rhs==1 or equal → lhs.
        const int64_t expected = (lhs_dim == 1)                         ? rhs_dim
                                 : (rhs_dim == 1 || lhs_dim == rhs_dim) ? lhs_dim
                                                                        : int64_t{-1};
        if (expected < 0 || out_dim != expected) return false;
    }
    return true;
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
                    std::string("CpuAddKernel ") + name + " offset overflow");
        }
        int64_t new_max = 0;
        if (CheckOverflowAdd(max_offset, contrib, &new_max)) {
            return Status::InvalidArgument(
                    std::string("CpuAddKernel ") + name + " offset overflow");
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
                    "CpuAddKernel output element count overflow");
        }
        if (next < 0) {
            return Status::InvalidArgument(
                    "CpuAddKernel output element count exceeds int64_t");
        }
        count = next;
    }
    return count;
}

Status ValidateAndExecute(const CpuAddParams* params) noexcept {
    const TensorView& lhs = params->lhs_tensor;
    const TensorView& rhs = params->rhs_tensor;
    const MutableTensorView& output = params->output_tensor;

    if (!lhs.is_valid()) {
        return Status::InvalidArgument("CpuAddKernel requires a valid lhs TensorView");
    }
    if (!rhs.is_valid()) {
        return Status::InvalidArgument("CpuAddKernel requires a valid rhs TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument("CpuAddKernel requires a valid output MutableTensorView");
    }

    const DataType dtype = lhs.dtype();
    if (rhs.dtype() != dtype || output.dtype() != dtype) {
        return Status::InvalidArgument(
                "CpuAddKernel requires matching lhs, rhs, and output dtypes");
    }
    if (!IsSupportedAddDType(dtype)) {
        return Status::InvalidArgument(
                "CpuAddKernel only supports float32, float64, bfloat16, int32, and int64 tensors");
    }

    const int32_t output_rank = output.rank();
    const int32_t expected_rank = std::max(lhs.rank(), rhs.rank());
    if (output_rank != expected_rank) {
        return Status::InvalidArgument(
                "CpuAddKernel output rank must equal max(lhs rank, rhs rank)");
    }
    if (output_rank > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument("CpuAddKernel output rank exceeds maximum supported rank");
    }

    if (!ValidateBroadcastCompatible(lhs.shape(), rhs.shape(), output.shape())) {
        return Status::InvalidArgument(
                "CpuAddKernel input shapes are not broadcast-compatible with output shape");
    }

    const auto numel_or = CheckedOutputNumel(output_rank, output.shape());
    if (!numel_or.ok()) return numel_or.status();
    const int64_t numel = numel_or.value();
    if (numel == 0) {
        return Status::Ok();
    }

    if (lhs.data() == nullptr) {
        return Status::InvalidArgument("CpuAddKernel requires non-null lhs data");
    }
    if (rhs.data() == nullptr) {
        return Status::InvalidArgument("CpuAddKernel requires non-null rhs data");
    }
    if (output.data() == nullptr) {
        return Status::InvalidArgument("CpuAddKernel requires non-null output data");
    }

    {
        auto status = ValidateMaxOffset(lhs.rank(), lhs.shape(), lhs.strides(), "lhs");
        if (!status.ok()) return status;
        status = ValidateMaxOffset(rhs.rank(), rhs.shape(), rhs.strides(), "rhs");
        if (!status.ok()) return status;
        status = ValidateMaxOffset(output.rank(), output.shape(), output.strides(), "output");
        if (!status.ok()) return status;
    }

    return cpu::detail::AddKernel_CPU_Scalar(lhs, rhs, output, numel);
}

}// namespace

Status CpuAddKernel(const KernelContext& ctx) noexcept {
    const CpuAddParams* params = GetParams(ctx.kernel_params);
    if (params == nullptr) {
        return Status::InvalidArgument(
                "CpuAddKernel requires CpuAddParams in KernelContext.kernel_params");
    }

    return ValidateAndExecute(params);
}

AM_REGISTER_KERNEL(CpuAddFp32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType::Float32(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuAddKernel,
                           .name = "cpu::add_f32_scalar",
                           .priority = 10,
                   })

AM_REGISTER_KERNEL(CpuAddFp64Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Double(),
                                   .weight_dtype = DataType::Double(),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuAddKernel,
                           .name = "cpu::add_f64_scalar",
                           .priority = 10,
                   })

AM_REGISTER_KERNEL(CpuAddBf16Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::BFloat(16),
                                   .weight_dtype = DataType::BFloat(16),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuAddKernel,
                           .name = "cpu::add_bf16_scalar",
                           .priority = 10,
                   })

AM_REGISTER_KERNEL(CpuAddI32Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Int(32),
                                   .weight_dtype = DataType::Int(32),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuAddKernel,
                           .name = "cpu::add_i32_scalar",
                           .priority = 10,
                   })

AM_REGISTER_KERNEL(CpuAddI64Scalar,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Int(64),
                                   .weight_dtype = DataType::Int(64),
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuAddKernel,
                           .name = "cpu::add_i64_scalar",
                           .priority = 10,
                   })

}// namespace aethermind
