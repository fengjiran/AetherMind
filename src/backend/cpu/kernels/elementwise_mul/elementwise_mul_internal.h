/// Internal declarations for the CPU ElementwiseMul kernel.
///
/// Declares the backend-internal kernel entry point (ElementwiseMulKernel),
/// its compute-ready args struct (ElementwiseMulKernelArgs), and the scalar
/// broadcast micro-kernel. Operator code never includes this header; the
/// KernelParamsBuilder indirection keeps operators free of backend
/// internals.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H

#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/shape_and_stride.h"

#include <array>

namespace aethermind::cpu::detail {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

/// Compute-ready args for the CPU ElementwiseMul kernel.
///
/// Produced by the `KernelParamsBuilder` registered with this kernel
/// (BuildElementwiseMulArgs in elementwise_mul_entry.cpp) and consumed by
/// the scalar broadcast micro-kernel. `numel` is the broadcast output
/// element count; a zero count means the kernel returns before dispatch.
struct ElementwiseMulKernelArgs {
    const float* lhs_data{};
    const float* rhs_data{};
    float* output_data{};
    int64_t numel{};
    int32_t lhs_rank{};
    int32_t rhs_rank{};
    int32_t output_rank{};
    std::array<int64_t, kMaxRank> lhs_shape{};
    std::array<int64_t, kMaxRank> lhs_strides{};
    std::array<int64_t, kMaxRank> rhs_shape{};
    std::array<int64_t, kMaxRank> rhs_strides{};
    std::array<int64_t, kMaxRank> output_shape{};
    std::array<int64_t, kMaxRank> output_strides{};
};

/// Runs the scalar FP32 ElementwiseMul broadcast micro-kernel.
///
/// @param args Pre-validated kernel arguments. Data pointers must be
///        non-null when `numel` is positive.
/// @return Ok on success.
Status RunElementwiseMulScalar(const ElementwiseMulKernelArgs& args) noexcept;

/// Kernel entry point registered via KernelDescriptor::kernel_func.
Status ElementwiseMulKernel(const KernelContext& ctx) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H