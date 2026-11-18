#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ATTENTION_ATTENTION_AVX2_INTERNAL_H

#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "attention_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

// All helpers in this header are local to the ISA translation unit.
namespace aethermind::cpu::detail {
namespace {

enum class KVElement { kF32,
                       kF16,
                       kBF16 };

template<KVElement Element>
constexpr size_t kElementBytes = Element == KVElement::kF32 ? sizeof(float) : sizeof(uint16_t);

template<KVElement Element>
inline float LoadScalar(const std::byte* data) noexcept {
    if constexpr (Element == KVElement::kF32) {
        float value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    } else {
        uint16_t bits;
        std::memcpy(&bits, data, sizeof(bits));
        if constexpr (Element == KVElement::kF16) {
            return Half(bits, Half::from_bits());
        } else {
            return BFloat16(bits, BFloat16::from_bits());
        }
    }
}

template<KVElement Element>
inline __m256 LoadEight(const std::byte* data) noexcept {
    if constexpr (Element == KVElement::kF32) {
        __m256 values;
        std::memcpy(&values, data, sizeof(values));
        return values;
    } else {
        __m128i bits;
        std::memcpy(&bits, data, sizeof(bits));
        constexpr int16_t exponent = Element == KVElement::kF16 ? 0x7C00 : 0x7F80;
        constexpr int16_t fraction = Element == KVElement::kF16 ? 0x03FF : 0x007F;
        constexpr int16_t quiet_bit = Element == KVElement::kF16 ? 0x0200 : 0x0040;
        const __m128i special = _mm_cmpeq_epi16(_mm_and_si128(bits, _mm_set1_epi16(exponent)),
                                                _mm_set1_epi16(exponent));
        const __m128i zero_fraction = _mm_cmpeq_epi16(_mm_and_si128(bits, _mm_set1_epi16(fraction)),
                                                      _mm_setzero_si128());
        // Match the scalar widening contract before performing FP arithmetic.
        bits = _mm_or_si128(bits, _mm_and_si128(_mm_andnot_si128(zero_fraction, special),
                                                _mm_set1_epi16(quiet_bit)));
        if constexpr (Element == KVElement::kF16) {
            return _mm256_cvtph_ps(bits);
        } else {
            return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
        }
    }
}

template<KVElement Element>
inline float Dot(const float* query, const std::byte* key, size_t dims) noexcept {
    float sum = 0.0F;
    size_t d = 0;
    for (; dims - d >= 8; d += 8) {
        const __m256 products = _mm256_mul_ps(_mm256_loadu_ps(query + d),
                                              LoadEight<Element>(key + d * kElementBytes<Element>));
        float lanes[8];
        _mm256_storeu_ps(lanes, products);
        // Keep the reference's ascending FP32 additions. A tree reduction can
        // change overflow/NaN classification as well as finite rounding.
        for (float product: lanes) {
            sum += product;
        }
    }
    for (; d < dims; ++d) {
        sum += query[d] * LoadScalar<Element>(key + d * kElementBytes<Element>);
    }
    return sum;
}

template<KVElement Element>
void AccumulateHead(const float* query, float* output, size_t dims, float scale,
                    const std::byte* keys, const std::byte* values, size_t token_stride,
                    size_t key_end) noexcept {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < key_end; ++token) {
        const float logit = Dot<Element>(query, keys + token * token_stride, dims) * scale;
        // Keep both the reference's NaN comparison order and this FP helper in the ISA TU.
        if (max_logit < logit) {
            max_logit = logit;
        }
    }

    std::fill_n(output, dims, 0.0F);
    float denominator = 0.0F;
    for (size_t token = 0; token < key_end; ++token) {
        const auto* value = values + token * token_stride;
        const float logit = Dot<Element>(query, keys + token * token_stride, dims) * scale;
        // Call the C overload directly so a Debug build cannot export an AVX-compiled
        // weak std::exp(float) definition that coalesces into the baseline path.
        const float weight = ::expf(logit - max_logit);
        denominator += weight;
        const __m256 weights = _mm256_set1_ps(weight);
        size_t d = 0;
        for (; dims - d >= 8; d += 8) {
            const __m256 products = _mm256_mul_ps(weights, LoadEight<Element>(value + d * kElementBytes<Element>));
            _mm256_storeu_ps(output + d, _mm256_add_ps(_mm256_loadu_ps(output + d), products));
        }
        for (; d < dims; ++d) {
            output[d] += weight * LoadScalar<Element>(value + d * kElementBytes<Element>);
        }
    }
    const __m256 norm = _mm256_set1_ps(denominator);
    size_t d = 0;
    for (; dims - d >= 8; d += 8) {
        _mm256_storeu_ps(output + d, _mm256_div_ps(_mm256_loadu_ps(output + d), norm));
    }
    for (; d < dims; ++d) {
        output[d] /= denominator;
    }
}

// A16 needs FP32 local storage for one widened Q and one accumulator. Larger
// heads retain the reference's fixed-size blocked implementation.
constexpr size_t kMaxA16HeadDim = 256;

template<KVElement Element, bool A16 = false, typename Args>
Status RunContiguousHeads(const Args& args, const KVCacheReadBinding& read,
                          size_t first_row = 0) noexcept {
    constexpr size_t act_bytes = A16 ? kElementBytes<Element> : sizeof(float);
    const auto dims = static_cast<size_t>(args.head_dim);
    const auto& storage = read.storage;
    const auto group_size = static_cast<size_t>(args.num_q_heads / args.num_kv_heads);
    for (size_t row = first_row; row < static_cast<size_t>(args.seq_len); ++row) {
        const size_t key_end = read.query_begin + row + 1;
        for (size_t head = 0; head < static_cast<size_t>(args.num_q_heads); ++head) {
            const size_t kv_offset = (head / group_size) * storage.head_stride_bytes;
            const auto* query = reinterpret_cast<const std::byte*>(args.query) +
                                (row * args.query_row_stride + head * dims) * act_bytes;
            auto* output = reinterpret_cast<std::byte*>(args.output) +
                           (row * args.output_row_stride + head * dims) * act_bytes;
            if constexpr (A16) {
                float widened[kMaxA16HeadDim];
                float accum[kMaxA16HeadDim];
                size_t d = 0;
                for (; dims - d >= 8; d += 8) {
                    _mm256_storeu_ps(widened + d, LoadEight<Element>(query + d * act_bytes));
                }
                for (; d < dims; ++d) {
                    widened[d] = LoadScalar<Element>(query + d * act_bytes);
                }
                AccumulateHead<Element>(widened, accum, dims, args.scale,
                                        storage.key_data + kv_offset, storage.value_data + kv_offset,
                                        storage.token_stride_bytes, key_end);
                for (d = 0; d < dims; ++d) {
                    StoreA16Element(output + d * act_bytes, accum[d], args.act_dtype);
                }
            } else {
                AccumulateHead<Element>(reinterpret_cast<const float*>(query),
                                        reinterpret_cast<float*>(output), dims, args.scale,
                                        storage.key_data + kv_offset, storage.value_data + kv_offset,
                                        storage.token_stride_bytes, key_end);
            }
        }
    }
    return Status::Ok();
}

bool HasVectorWidth(const auto& args) noexcept {
    return args.query_col_stride == 1 && args.output_col_stride == 1 && args.head_dim >= 8;
}

} // namespace
} // namespace aethermind::cpu::detail
#endif
