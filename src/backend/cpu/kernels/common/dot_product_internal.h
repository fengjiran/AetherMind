#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_INTERNAL_H

/// @file dot_product_internal.h
/// Internal declarations for the CPU FP32 dot-product cores.
///
/// Exposes the per-ISA cores so tests can exercise each implementation
/// directly and same-ISA kernels can call a core without re-entering the
/// public dispatch. The AVX2 core symbol only exists when the library is
/// built with AVX2/FMA enabled.

#include "aethermind/base/macros.h"

#include <cstddef>

namespace aethermind::cpu {

/// @brief Scalar core: single accumulator chain, no vectorization.
AM_NODISCARD float DotProductF32Scalar(const float* a, const float* b, std::size_t n) noexcept;

/// @brief Scalar multi-target core. Each target is one scalar dot product.
void DotProductF32MultiTargetScalar(const float* lhs, const float* rhs, std::size_t n,
                                    std::size_t num_targets, std::size_t rhs_stride,
                                    float* out) noexcept;

#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
/// @brief AVX2/FMA core: eight 8-wide accumulator chains over 64-element
///        blocks with graded 32/16/8-wide vector tails and a scalar tail.
AM_NODISCARD float DotProductF32Avx2(const float* a, const float* b, std::size_t n) noexcept;

/// @brief AVX2/FMA multi-target core: processes targets in 4-wide blocks
///        with two accumulator chains per target, falling back to the
///        single-target core for leftover targets.
void DotProductF32MultiTargetAvx2(const float* lhs, const float* rhs, std::size_t n,
                                  std::size_t num_targets, std::size_t rhs_stride,
                                  float* out) noexcept;
#endif

} // namespace aethermind::cpu

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_INTERNAL_H