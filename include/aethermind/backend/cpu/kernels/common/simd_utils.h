#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_SIMD_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_SIMD_UTILS_H

#include "aethermind/base/macros.h"

#include <array>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace aethermind {

#if defined(__AVX2__)
// NOLINTBEGIN(portability-simd-intrinsics)
// This helper is intentionally AVX2-specific; per-ISA kernels are the
// backend's design, so the portability check does not apply.
AM_NODISCARD AM_ALWAYS_INLINE float HorizontalSumAvx2(__m256 v) noexcept {
    const __m128 vlow = _mm256_castps256_ps128(v);
    const __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum = _mm_add_ps(vhigh, vlow);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}
// NOLINTEND(portability-simd-intrinsics)
#endif

#if defined(__AVX512F__)
// NOLINTBEGIN(portability-simd-intrinsics)
// This helper is intentionally AVX-512-specific; per-ISA kernels are the
// backend's design, so the portability check does not apply.
AM_NODISCARD AM_ALWAYS_INLINE float ReduceLanes(__m512 v) noexcept {
    alignas(64) std::array<float, 16> lanes{};
    _mm512_store_ps(lanes.data(), v);
    float sum = 0.0F;
    for (const float lane: lanes) {
        sum = sum + lane;
    }
    return sum;
}
// NOLINTEND(portability-simd-intrinsics)
#endif

} // namespace aethermind

#endif
