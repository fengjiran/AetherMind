#include "aethermind/backend/cpu/kernels/cpu_dot_product_avx2.h"
#include "aethermind/backend/cpu/kernels/common/cpu_simd_utils.h"

#include <immintrin.h>

namespace aethermind {

float DotProductAvx2Unroll(const float* a, const float* b, std::size_t n) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vsum0);
        vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), vsum1);
        vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), vsum2);
        vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), vsum3);
    }

    __m256 vres = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1), _mm256_add_ps(vsum2, vsum3));
    for (; i + 8 <= n; i += 8) {
        vres = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vres);
    }

    float result = HorizontalSumAvx2(vres);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#else
    float result = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

}// namespace aethermind
