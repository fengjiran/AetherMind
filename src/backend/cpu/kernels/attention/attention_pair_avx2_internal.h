#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_PAIR_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_PAIR_AVX2_INTERNAL_H

#include "attention_avx2_internal.h"

namespace aethermind::cpu::detail {
namespace {

struct HeadPair {
    float first{};
    float second{};
};

template<KVElement Element>
inline HeadPair DotPair(const float* q0, const float* q1, const std::byte* key, size_t dims) noexcept {
    HeadPair sum;
    size_t d = 0;
    for (; dims - d >= 8; d += 8) {
        const __m256 k = LoadEight<Element>(key + d * kElementBytes<Element>);
        float products0[8];
        float products1[8];
        _mm256_storeu_ps(products0, _mm256_mul_ps(_mm256_loadu_ps(q0 + d), k));
        _mm256_storeu_ps(products1, _mm256_mul_ps(_mm256_loadu_ps(q1 + d), k));
        // Interleave two independent chains without changing either head's order.
        for (size_t lane = 0; lane < 8; ++lane) {
            sum.first += products0[lane];
            sum.second += products1[lane];
        }
    }
    for (; d < dims; ++d) {
        const float k = LoadScalar<Element>(key + d * kElementBytes<Element>);
        sum.first += q0[d] * k;
        sum.second += q1[d] * k;
    }
    return sum;
}

// ExtraKeyForSecondQuery is used only for consecutive Prefill rows. The
// caller supplies the first row's exclusive end and validates both rows.
template<KVElement Element, bool ExtraKeyForSecondQuery = false>
void AccumulateHeadPair(const float* q0, const float* q1, float* out0, float* out1,
                        size_t dims, float scale, const std::byte* keys,
                        const std::byte* values, size_t token_stride, size_t key_end) noexcept {
    constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
    HeadPair max{kNegativeInfinity, kNegativeInfinity};
    for (size_t token = 0; token < key_end; ++token) {
        const auto dot = DotPair<Element>(q0, q1, keys + token * token_stride, dims);
        const float logit0 = dot.first * scale;
        const float logit1 = dot.second * scale;
        if (max.first < logit0) {
            max.first = logit0;
        }
        if (max.second < logit1) {
            max.second = logit1;
        }
    }

    float tail_logit = 0.0F;
    if constexpr (ExtraKeyForSecondQuery) {
        tail_logit = Dot<Element>(q1, keys + key_end * token_stride, dims) * scale;
        if (max.second < tail_logit) {
            max.second = tail_logit;
        }
    }

    std::fill_n(out0, dims, 0.0F);
    std::fill_n(out1, dims, 0.0F);
    HeadPair denominator;
    for (size_t token = 0; token < key_end; ++token) {
        const auto dot = DotPair<Element>(q0, q1, keys + token * token_stride, dims);
        const float weight0 = ::expf(dot.first * scale - max.first);
        const float weight1 = ::expf(dot.second * scale - max.second);
        denominator.first += weight0;
        denominator.second += weight1;
        const __m256 weights0 = _mm256_set1_ps(weight0);
        const __m256 weights1 = _mm256_set1_ps(weight1);
        const auto* value = values + token * token_stride;
        size_t d = 0;
        for (; dims - d >= 8; d += 8) {
            const __m256 v = LoadEight<Element>(value + d * kElementBytes<Element>);
            _mm256_storeu_ps(out0 + d, _mm256_add_ps(_mm256_loadu_ps(out0 + d), _mm256_mul_ps(weights0, v)));
            _mm256_storeu_ps(out1 + d, _mm256_add_ps(_mm256_loadu_ps(out1 + d), _mm256_mul_ps(weights1, v)));
        }
        for (; d < dims; ++d) {
            const float v = LoadScalar<Element>(value + d * kElementBytes<Element>);
            out0[d] += weight0 * v;
            out1[d] += weight1 * v;
        }
    }

    if constexpr (ExtraKeyForSecondQuery) {
        // A masked value must not be read or multiplied by zero: it may be NaN.
        const float weight = ::expf(tail_logit - max.second);
        denominator.second += weight;
        const __m256 weights = _mm256_set1_ps(weight);
        const auto* value = values + key_end * token_stride;
        size_t d = 0;
        for (; dims - d >= 8; d += 8) {
            const __m256 v = LoadEight<Element>(value + d * kElementBytes<Element>);
            _mm256_storeu_ps(out1 + d, _mm256_add_ps(_mm256_loadu_ps(out1 + d), _mm256_mul_ps(weights, v)));
        }
        for (; d < dims; ++d) {
            out1[d] += weight * LoadScalar<Element>(value + d * kElementBytes<Element>);
        }
    }

    const __m256 norm0 = _mm256_set1_ps(denominator.first);
    const __m256 norm1 = _mm256_set1_ps(denominator.second);
    size_t d = 0;
    for (; dims - d >= 8; d += 8) {
        _mm256_storeu_ps(out0 + d, _mm256_div_ps(_mm256_loadu_ps(out0 + d), norm0));
        _mm256_storeu_ps(out1 + d, _mm256_div_ps(_mm256_loadu_ps(out1 + d), norm1));
    }
    for (; d < dims; ++d) {
        out0[d] /= denominator.first;
        out1[d] /= denominator.second;
    }
}

template<KVElement Element>
void WidenQueryHead(const std::byte* query, float* widened, size_t dims) noexcept {
    size_t d = 0;
    for (; dims - d >= 8; d += 8) {
        _mm256_storeu_ps(widened + d, LoadEight<Element>(query + d * kElementBytes<Element>));
    }
    for (; d < dims; ++d) {
        widened[d] = LoadScalar<Element>(query + d * kElementBytes<Element>);
    }
}

} // namespace
} // namespace aethermind::cpu::detail
#endif
