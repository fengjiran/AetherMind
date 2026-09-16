#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ACTIVATION_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ACTIVATION_UTILS_H

/// @file activation_utils.h
/// @brief Numerically stable scalar activation helpers for CPU reference kernels.

#include <cmath>

namespace aethermind::cpu::detail {

/// Computes SiLU(x) = x * sigmoid(x) without overflowing exp for finite x.
inline float ComputeSiluF32(float value) noexcept {
    if (value >= 0.0F) {
        return value / (1.0F + std::exp(-value));
    }

    const float exp_value = std::exp(value);
    return value * exp_value / (1.0F + exp_value);
}

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ACTIVATION_UTILS_H
