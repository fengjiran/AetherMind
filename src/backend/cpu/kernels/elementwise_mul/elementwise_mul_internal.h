/// Internal declarations for the CPU ElementwiseMul kernel.
///
/// Defines the pre-validated FP32 compute-ready args struct
/// (ElementwiseMulF32KernelArgs) and the reference broadcast micro-kernel.
/// The kernel entry itself is TU-local to elementwise_mul_entry.cpp;
/// operators never include this header.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H

#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/base/status.h"

#include <array>

namespace aethermind::cpu::detail {

/// Compute-ready args for the CPU ElementwiseMul kernel.
///
/// Produced by the `KernelParamsBuilder` registered with this kernel
/// (BuildElementwiseMulF32ReferenceArgs in elementwise_mul_entry.cpp) and consumed by
/// the reference broadcast micro-kernel. `numel` is the broadcast output
/// element count; a zero count means the kernel returns before dispatch.
struct ElementwiseMulF32KernelArgs {
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

/// Runs the reference FP32 ElementwiseMul broadcast micro-kernel.
///
/// @param args Pre-validated kernel arguments. Data pointers must be
///        non-null when `numel` is positive.
/// @return Ok on success.
Status RunElementwiseMulF32Reference(const ElementwiseMulF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ELEMENTWISE_MUL_INTERNAL_H