#ifndef AETHERMIND_BACKEND_CPU_KERNELS_SILU_SILU_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_SILU_SILU_INTERNAL_H

/// @file silu_internal.h
/// @brief Backend-internal FP32 SiLU compute-ready parameters and micro-kernel.

#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/base/status.h"

#include <array>

namespace aethermind::cpu::detail {

/// Pre-validated FP32 arguments for the SiLU reference micro-kernel.
///
/// The binding-time builder validates matching input/output shapes, address
/// arithmetic, an injective output mapping, and the exact-in-place alias
/// exception before these values are consumed on the execution hot path.
struct SiluF32KernelArgs {
    const float* input_data{};
    float* output_data{};
    int64_t numel{};
    bool is_flat{};
    int32_t rank{};
    std::array<int64_t, kMaxRank> shape{};
    std::array<int64_t, kMaxRank> input_strides{};
    std::array<int64_t, kMaxRank> output_strides{};
};

/// Runs the FP32, stride-aware SiLU reference micro-kernel.
Status RunSiluF32Reference(const SiluF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_SILU_SILU_INTERNAL_H
