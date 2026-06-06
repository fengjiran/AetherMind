#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_SIMD_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_SIMD_UTILS_H

#include "macros.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace aethermind {

#if defined(__AVX2__)
AM_NODISCARD AM_ALWAYS_INLINE float HorizontalSumAvx2(__m256 v) noexcept {
    const __m128 vlow = _mm256_castps256_ps128(v);
    const __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum = _mm_add_ps(vhigh, vlow);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}
#endif

}// namespace aethermind

#endif
