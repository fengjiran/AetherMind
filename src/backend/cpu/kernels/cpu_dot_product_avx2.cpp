#include "aethermind/backend/cpu/kernels/cpu_dot_product_avx2.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind {
namespace {

#if defined(__AVX2__) && defined(__FMA__)
float HorizontalSumAvx2(__m256 v) noexcept {
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(high, low);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

}// namespace

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
