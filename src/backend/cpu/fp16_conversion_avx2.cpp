#include "aethermind/dtypes/fp16_conversion.h"
#include "fp16_conversion_internal.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace aethermind::cpu {

void ConvertFp32ToFp16Avx2(const std::byte* src, std::byte* dst, size_t count) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i fraction_mask = _mm256_set1_epi32(0x007FFFFF);
    size_t index = 0;
    for (; count - index >= 8; index += 8) {
        __m256i bits;
        std::memcpy(&bits, src + index * sizeof(float), sizeof(bits));
        const __m256i sign = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x8000));
        const __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
        const __m256i fraction = _mm256_and_si256(bits, fraction_mask);

        __m256i normal = _mm256_or_si256(
                _mm256_slli_epi32(_mm256_sub_epi32(exponent, _mm256_set1_epi32(112)), 10),
                _mm256_srli_epi32(fraction, 13));
        const __m256i normal_remainder = _mm256_and_si256(fraction, _mm256_set1_epi32(0x1FFF));
        const __m256i normal_halfway = _mm256_set1_epi32(0x1000);
        const __m256i normal_round = _mm256_or_si256(
                _mm256_cmpgt_epi32(normal_remainder, normal_halfway),
                _mm256_and_si256(_mm256_cmpeq_epi32(normal_remainder, normal_halfway),
                                 _mm256_cmpeq_epi32(_mm256_and_si256(normal, one), one)));
        normal = _mm256_add_epi32(normal, _mm256_and_si256(normal_round, one));

        // Every lane executes these shifts, including lanes later discarded.
        const __m256i shift = _mm256_min_epi32(_mm256_set1_epi32(24),
                                               _mm256_max_epi32(_mm256_set1_epi32(14),
                                                                _mm256_sub_epi32(_mm256_set1_epi32(126), exponent)));
        const __m256i significand = _mm256_or_si256(fraction, _mm256_set1_epi32(0x00800000));
        __m256i subnormal = _mm256_srlv_epi32(significand, shift);
        const __m256i remainder_mask = _mm256_sub_epi32(_mm256_sllv_epi32(one, shift), one);
        const __m256i remainder = _mm256_and_si256(significand, remainder_mask);
        const __m256i halfway = _mm256_sllv_epi32(one, _mm256_sub_epi32(shift, one));
        const __m256i subnormal_round = _mm256_or_si256(
                _mm256_cmpgt_epi32(remainder, halfway),
                _mm256_and_si256(_mm256_cmpeq_epi32(remainder, halfway),
                                 _mm256_cmpeq_epi32(_mm256_and_si256(subnormal, one), one)));
        subnormal = _mm256_add_epi32(subnormal, _mm256_and_si256(subnormal_round, one));

        const __m256i is_normal = _mm256_and_si256(_mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(112)),
                                                   _mm256_cmpgt_epi32(_mm256_set1_epi32(143), exponent));
        const __m256i is_subnormal = _mm256_and_si256(_mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(101)),
                                                      _mm256_cmpgt_epi32(_mm256_set1_epi32(113), exponent));
        __m256i magnitude = _mm256_or_si256(_mm256_and_si256(normal, is_normal),
                                            _mm256_and_si256(subnormal, is_subnormal));
        const __m256i infinity = _mm256_set1_epi32(0x7C00);
        magnitude = _mm256_blendv_epi8(magnitude, infinity,
                                       _mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(142)));
        const __m256i is_nan = _mm256_andnot_si256(
                _mm256_cmpeq_epi32(fraction, zero),
                _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(255)));
        const __m256i nan = _mm256_or_si256(_mm256_set1_epi32(0x7E00), _mm256_srli_epi32(fraction, 13));
        magnitude = _mm256_blendv_epi8(magnitude, nan, is_nan);
        const __m256i result = _mm256_or_si256(sign, magnitude);

        // packus works independently inside 128-bit lanes. Reorder qwords so
        // the low half contains elements 0..7, including their stored sign bits.
        const __m256i packed = _mm256_packus_epi32(result, zero);
        const __m128i output = _mm256_castsi256_si128(_mm256_permute4x64_epi64(packed, 0xD8));
        std::memcpy(dst + index * sizeof(uint16_t), &output, sizeof(output));
    }
    if (index < count) {
        ConvertFp32ToFp16Scalar(src + index * sizeof(float),
                                dst + index * sizeof(uint16_t), count - index);
    }
}

} // namespace aethermind::cpu
