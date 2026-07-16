// Kernel entry for the CPU Add operator (directory-structured).
//
// Validates CpuAddParams, computes the broadcast output shape via
// BroadcastShapes, and dispatches to the scalar float32 implementation
// (AddKernel_CPU_FP32_Scalar) defined in add_fp32_scalar.cpp. Kernel
// registration with KernelSelector{weight_dtype=DataType{}} lives here via
// AM_REGISTER_KERNEL, allowing the new directory-structured kernel to
// coexist with the legacy cpu_add_kernel.cpp registration that uses
// weight_dtype=DataType::Float32(): the registry keys differ, so both can
// be registered and resolved independently.

#include "add_internal.h"

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/shape_inference/broadcast.h"
#include "aethermind/utils/overflow_check.h"

#include <cstdint>
#include <span>

namespace aethermind::cpu {

namespace {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

const CpuAddParams* GetParams(const void* kernel_params) noexcept {
    return static_cast<const CpuAddParams*>(kernel_params);
}

// Computes the total element count of `shape` with overflow checking.
// rank-0 (empty span) returns 1; any zero dimension returns 0.
StatusOr<int64_t> CheckedNumel(std::span<const int64_t> shape) noexcept {
    if (shape.empty()) return int64_t{1};

    int64_t count = 1;
    for (int64_t dim : shape) {
        if (dim == 0) return int64_t{0};
        int64_t next = 0;
        if (CheckOverflowMul(count, dim, &next)) {
            return Status::InvalidArgument(
                    "CpuAddKernel output element count overflow");
        }
        count = next;
    }
    return count;
}

// Validates ctx against CpuAddParams requirements and returns the output
// element count on success. A count of 0 means the caller should return Ok
// without dispatching to the execution path (no data access).
StatusOr<int64_t> ValidateAddEntry(const KernelContext& ctx) noexcept {
    const CpuAddParams* params = GetParams(ctx.kernel_params);
    if (params == nullptr) {
        return Status::InvalidArgument(
                "CpuAddKernel requires CpuAddParams in KernelContext.kernel_params");
    }

    const TensorView& lhs = params->lhs;
    const TensorView& rhs = params->rhs;
    const MutableTensorView& output = params->output;

    if (!lhs.is_valid()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires a valid lhs TensorView");
    }
    if (!rhs.is_valid()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires a valid rhs TensorView");
    }
    if (!output.is_valid()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires a valid output MutableTensorView");
    }

    if (lhs.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires float32 lhs TensorView");
    }
    if (rhs.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires float32 rhs TensorView");
    }
    if (output.dtype() != DataType::Float32()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires float32 output MutableTensorView");
    }

    // BroadcastShapes is the single source of truth for broadcast
    // compatibility and the resolved output shape. Not a hot path, so the
    // std::vector allocation is acceptable here.
    auto broadcast_or = BroadcastShapes(lhs.shape(), rhs.shape());
    if (!broadcast_or.ok()) {
        return broadcast_or.status();
    }
    const std::vector<int64_t>& expected_shape = broadcast_or.value();

    if (output.rank() != static_cast<int32_t>(expected_shape.size())) {
        return Status::InvalidArgument(
                "CpuAddKernel output rank must equal broadcast result rank");
    }
    if (output.rank() > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument(
                "CpuAddKernel output rank exceeds maximum supported rank");
    }
    for (int32_t i = 0; i < output.rank(); ++i) {
        if (output.dim(i) != expected_shape[i]) {
            return Status::InvalidArgument(
                    "CpuAddKernel output shape must equal broadcast result shape");
        }
    }

    // The scalar execution path requires a contiguous output so it can use
    // flat indexing into output_data. The caller is responsible for
    // materializing a contiguous output buffer.
    if (!output.is_contiguous()) {
        return Status::InvalidArgument(
                "CpuAddKernel requires contiguous output");
    }

    auto numel_or = CheckedNumel(output.shape());
    if (!numel_or.ok()) return numel_or.status();
    const int64_t numel = numel_or.value();

    // For zero-element outputs, return early without touching data pointers:
    // callers may pass null data for empty tensors.
    if (numel == 0) {
        return int64_t{0};
    }

    // With numel > 0, data pointers must be non-null.
    if (lhs.data() == nullptr) {
        return Status::InvalidArgument(
                "CpuAddKernel requires non-null lhs data");
    }
    if (rhs.data() == nullptr) {
        return Status::InvalidArgument(
                "CpuAddKernel requires non-null rhs data");
    }
    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                "CpuAddKernel requires non-null output data");
    }

    return numel;
}

}// namespace

Status CpuAddKernelEntry_FP32_Scalar(const KernelContext& ctx) noexcept {
    auto numel_or = ValidateAddEntry(ctx);
    if (!numel_or.ok()) return numel_or.status();
    const int64_t numel = numel_or.value();
    if (numel == 0) return Status::Ok();

    const CpuAddParams* params = GetParams(ctx.kernel_params);
    return AddKernel_CPU_FP32_Scalar(params->lhs, params->rhs, params->output,
                                     numel);
}

AM_REGISTER_KERNEL(CpuAddFp32ScalarDirectory,
                   KernelDescriptor{
                           .op_type = OpType::kAdd,
                           .selector = KernelSelector{
                                   .device_type = DeviceType::kCPU,
                                   .act_dtype = DataType::Float32(),
                                   .weight_dtype = DataType{},
                                   .weight_format = WeightFormat::kPlain,
                                   .isa = IsaLevel::kScalar,
                                   .phase = ExecPhase::kBoth,
                           },
                           .kernel_func = &CpuAddKernelEntry_FP32_Scalar,
                           .name = "cpu::add_f32_scalar_v2",
                           .priority = 10,
                   });

}// namespace aethermind::cpu
