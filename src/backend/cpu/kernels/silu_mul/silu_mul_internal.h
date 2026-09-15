#ifndef AETHERMIND_BACKEND_CPU_KERNELS_SILU_MUL_SILU_MUL_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_SILU_MUL_SILU_MUL_INTERNAL_H

/// @file silu_mul_internal.h
/// @brief Backend-internal FP32 SiluMul compute-ready parameters and micro-kernel.

#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/base/status.h"

#include <array>

namespace aethermind::cpu::detail {

/// Pre-validated FP32 arguments for the broadcast-aware SiluMul reference kernel.
///
/// The binding-time builder validates NumPy-style broadcasting, output
/// injectivity, address arithmetic, and alias safety before this structure is
/// executed without allocation or registry lookup. Operands carry their
/// semantic names: gate is activated, up is multiplied directly.
struct SiluMulF32KernelArgs {
    const float* gate_data{};
    const float* up_data{};
    float* output_data{};
    int64_t numel{};
    bool is_flat{};
    int32_t gate_rank{};
    int32_t up_rank{};
    int32_t output_rank{};
    std::array<int64_t, kMaxRank> gate_shape{};
    std::array<int64_t, kMaxRank> gate_strides{};
    std::array<int64_t, kMaxRank> up_shape{};
    std::array<int64_t, kMaxRank> up_strides{};
    std::array<int64_t, kMaxRank> output_shape{};
    std::array<int64_t, kMaxRank> output_strides{};
};

/// Runs the FP32, broadcast- and stride-aware SiluMul reference micro-kernel.
Status RunSiluMulF32Reference(const SiluMulF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_SILU_MUL_SILU_MUL_INTERNAL_H
