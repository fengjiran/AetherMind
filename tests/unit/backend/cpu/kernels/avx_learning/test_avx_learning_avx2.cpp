#include "test_avx_learning_helpers.h"

#if !defined(__AVX2__) || !defined(__FMA__)
#error "AVX learning helpers must be compiled with AVX2 and FMA enabled"
#endif

#include <immintrin.h>

namespace aethermind::test::avx_learning {

void Add(const float* a, const float* b, float* out, std::size_t n) noexcept {
    constexpr std::size_t kStep = 8;
    std::size_t i = 0;
    while (i + kStep <= n) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
        i += kStep;
    }
    while (i < n) {
        out[i] = a[i] + b[i];
        ++i;
    }
}

void FusedMultiplyAdd(const float* a, const float* b, const float* c,
                      float* out, std::size_t n) noexcept {
    constexpr std::size_t kStep = 8;
    std::size_t i = 0;
    while (i + kStep <= n) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vb, vc));
        i += kStep;
    }
    while (i < n) {
        out[i] = a[i] * b[i] + c[i];
        ++i;
    }
}

void Broadcast8(float value, float* out) noexcept {
    _mm256_storeu_ps(out, _mm256_set1_ps(value));
}

float HorizontalSum8(const float* data) noexcept {
    const __m256 value = _mm256_loadu_ps(data);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    const __m128 lo = _mm256_castps256_ps128(value);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

float HorizontalMax8(const float* data) noexcept {
    const __m256 value = _mm256_loadu_ps(data);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    const __m128 lo = _mm256_castps256_ps128(value);
    __m128 maximum = _mm_max_ps(lo, hi);
    maximum = _mm_max_ps(maximum, _mm_shuffle_ps(maximum, maximum, _MM_SHUFFLE(2, 3, 0, 1)));
    maximum = _mm_max_ps(maximum, _mm_shuffle_ps(maximum, maximum, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(maximum);
}

void ShuffleWithinLane8(const float* data, float* out) noexcept {
    const __m256 value = _mm256_loadu_ps(data);
    _mm256_storeu_ps(out, _mm256_shuffle_ps(value, value, _MM_SHUFFLE(2, 3, 0, 1)));
}

void Permute8(const float* data, const int32_t* indices, float* out) noexcept {
    const __m256 value = _mm256_loadu_ps(data);
    const __m256i index = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
    _mm256_storeu_ps(out, _mm256_permutevar8x32_ps(value, index));
}

void ClampNegativeToZero8(const float* data, float* out) noexcept {
    const __m256 value = _mm256_loadu_ps(data);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 mask = _mm256_cmp_ps(value, zero, _CMP_LT_OS);
    _mm256_storeu_ps(out, _mm256_blendv_ps(value, zero, mask));
}

float SumSquares8(const float* data) noexcept {
    const __m256 value = _mm256_loadu_ps(data);
    const __m256 square = _mm256_mul_ps(value, value);
    const __m128 hi = _mm256_extractf128_ps(square, 1);
    const __m128 lo = _mm256_castps256_ps128(square);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void ScaleRmsNorm8(const float* row, const float* weight, float inv_rms, float* out) noexcept {
    const __m256 value = _mm256_loadu_ps(row);
    const __m256 scale = _mm256_set1_ps(inv_rms);
    const __m256 scaled = _mm256_mul_ps(value, scale);
    _mm256_storeu_ps(out, _mm256_mul_ps(scaled, _mm256_loadu_ps(weight)));
}

void Gather8(const float* table, const int32_t* indices, float* out) noexcept {
    const __m256i index = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
    _mm256_storeu_ps(out, _mm256_i32gather_ps(table, index, 4));
}

void SquareWithScalarTail(const float* data, float* out, std::size_t n) noexcept {
    constexpr std::size_t kStep = 8;
    std::size_t i = 0;
    while (i + kStep <= n) {
        const __m256 value = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(value, value));
        i += kStep;
    }
    while (i < n) {
        out[i] = data[i] * data[i];
        ++i;
    }
}

void MaskedLoadTwo(const float* data, float* out) noexcept {
    const __m128i mask = _mm_setr_epi32(-1, -1, 0, 0);
    _mm_storeu_ps(out, _mm_maskload_ps(data, mask));
}

} // namespace aethermind::test::avx_learning
