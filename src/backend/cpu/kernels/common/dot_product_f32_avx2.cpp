#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "dot_product_internal.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind::cpu {
namespace {

#if defined(__AVX2__) && defined(__FMA__)

// Lane width of one accumulator chain. Every vector stage below the main
// block is a multiple of this, so each tail stage executes at most once.
constexpr std::size_t kLanes = 8;
constexpr std::size_t kMainBlock = 64; // 8 chains x 8 lanes
constexpr std::size_t kTail32 = 32;
constexpr std::size_t kTail16 = 16;

#endif

} // namespace

#if defined(__AVX2__) && defined(__FMA__)

float DotProductF32Avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + kMainBlock <= n; i += kMainBlock) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
        sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32), sum4);
        sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40), sum5);
        sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48), sum6);
        sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56), sum7);
    }

    if (i + kTail32 <= n) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
        i += kTail32;
    }

    if (i + kTail16 <= n) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
        i += kTail16;
    }

    if (i + kLanes <= n) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        i += kLanes;
    }

    const __m256 total01 = _mm256_add_ps(_mm256_add_ps(sum0, sum1),
                                         _mm256_add_ps(sum2, sum3));
    const __m256 total23 = _mm256_add_ps(_mm256_add_ps(sum4, sum5),
                                         _mm256_add_ps(sum6, sum7));
    float result = HorizontalSumAvx2(_mm256_add_ps(total01, total23));
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void DotProductF32MultiTargetAvx2(const float* lhs, const float* rhs, std::size_t n,
                                  std::size_t num_targets, std::size_t rhs_stride,
                                  float* out) noexcept {
    // 4-target blocks share the two lhs vectors loaded per K iteration; the
    // eight accumulators (4 targets x 2 chains) plus the two lhs registers
    // fit the 16 YMM registers without spilling.
    std::size_t t = 0;
    for (; t + 4 <= num_targets; t += 4) {
        const float* r0 = rhs + t * rhs_stride;
        const float* r1 = r0 + rhs_stride;
        const float* r2 = r1 + rhs_stride;
        const float* r3 = r2 + rhs_stride;

        __m256 sum00 = _mm256_setzero_ps();
        __m256 sum01 = _mm256_setzero_ps();
        __m256 sum10 = _mm256_setzero_ps();
        __m256 sum11 = _mm256_setzero_ps();
        __m256 sum20 = _mm256_setzero_ps();
        __m256 sum21 = _mm256_setzero_ps();
        __m256 sum30 = _mm256_setzero_ps();
        __m256 sum31 = _mm256_setzero_ps();

        std::size_t k = 0;
        for (; k + 16 <= n; k += 16) {
            const __m256 lhs0 = _mm256_loadu_ps(lhs + k);
            const __m256 lhs1 = _mm256_loadu_ps(lhs + k + 8);
            sum00 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r0 + k), sum00);
            sum01 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(r0 + k + 8), sum01);
            sum10 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r1 + k), sum10);
            sum11 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(r1 + k + 8), sum11);
            sum20 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r2 + k), sum20);
            sum21 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(r2 + k + 8), sum21);
            sum30 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r3 + k), sum30);
            sum31 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(r3 + k + 8), sum31);
        }

        if (k + 8 <= n) {
            const __m256 lhs0 = _mm256_loadu_ps(lhs + k);
            sum00 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r0 + k), sum00);
            sum10 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r1 + k), sum10);
            sum20 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r2 + k), sum20);
            sum30 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(r3 + k), sum30);
            k += 8;
        }

        float o0 = HorizontalSumAvx2(_mm256_add_ps(sum00, sum01));
        float o1 = HorizontalSumAvx2(_mm256_add_ps(sum10, sum11));
        float o2 = HorizontalSumAvx2(_mm256_add_ps(sum20, sum21));
        float o3 = HorizontalSumAvx2(_mm256_add_ps(sum30, sum31));

        for (; k < n; ++k) {
            const float input = lhs[k];
            o0 += input * r0[k];
            o1 += input * r1[k];
            o2 += input * r2[k];
            o3 += input * r3[k];
        }

        out[t] = o0;
        out[t + 1] = o1;
        out[t + 2] = o2;
        out[t + 3] = o3;
    }

    for (; t < num_targets; ++t) {
        out[t] = DotProductF32Avx2(lhs, rhs + t * rhs_stride, n);
    }
}

#endif

} // namespace aethermind::cpu