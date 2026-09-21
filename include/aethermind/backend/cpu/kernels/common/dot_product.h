#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_H

/// @file dot_product.h
/// @brief CPU FP32 dot-product primitives shared across kernels.
///
/// Public entry points dispatch on the runtime-effective CPU feature set:
/// the AVX2/FMA core runs when the effective features contain both AVX2 and
/// FMA, otherwise the scalar core runs. Callers never need compile-time
/// feature knowledge and may include this header on any platform.

#include "aethermind/base/macros.h"

#include <cstddef>

namespace aethermind::cpu {

/// @brief Computes `result = sum(a[i] * b[i])` over `n` elements.
///
/// Returns `+0.0F` when `n == 0`. `a` and `b` must each point to at least
/// `n` readable floats; no alignment is required, and `a` and `b` may alias
/// each other.
AM_NODISCARD float DotProductF32(const float* a, const float* b, std::size_t n) noexcept;

/// @brief Computes `out[t] = sum_k lhs[k] * rhs[t * rhs_stride + k]` for
///        `t` in `[0, num_targets)`.
///
/// This is the rank-1-update / GEMM M=1 core shared with one shared `lhs`
/// row against `num_targets` K-contiguous `rhs` rows. `rhs_stride` is in
/// elements; each target row is `rhs + t * rhs_stride` and must hold `n`
/// readable floats (target rows may overlap reads). `out` must hold at
/// least `num_targets` writable floats. With `n == 0` every target is
/// written `+0.0F`; with `num_targets == 0` the call is a no-op.
void DotProductF32MultiTarget(const float* lhs, const float* rhs, std::size_t n,
                              std::size_t num_targets, std::size_t rhs_stride,
                              float* out) noexcept;

} // namespace aethermind::cpu

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_H