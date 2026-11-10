#include "aethermind/dtypes/bf16_conversion.h"
#include "bf16_conversion_internal.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace aethermind::cpu {

void ConvertBf16ToFp32Avx2(const std::byte* src, std::byte* dst, size_t count) noexcept {
    size_t index = 0;
    for (; count - index >= 16; index += 16) {
        __m256i bits;
        std::memcpy(&bits, src + index * sizeof(uint16_t), sizeof(bits));
        // Quiet NaNs in the 16-bit domain before widening, matching the scalar
        // contract: exponent all-ones and fraction non-zero forces the quiet bit.
        const __m256i exponent_all_ones = _mm256_cmpeq_epi16(
                _mm256_and_si256(bits, _mm256_set1_epi16(0x7F80)), _mm256_set1_epi16(0x7F80));
        const __m256i fraction_nonzero = _mm256_cmpeq_epi16(
                _mm256_and_si256(bits, _mm256_set1_epi16(0x007F)), _mm256_setzero_si256());
        const __m256i nan = _mm256_andnot_si256(fraction_nonzero, exponent_all_ones);
        bits = _mm256_or_si256(bits, _mm256_and_si256(nan, _mm256_set1_epi16(0x0040)));

        // bfloat16 occupies the high 16 bits of a binary32; widening is a
        // 16-bit left shift per element.
        const __m256i low = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(bits)), 16);
        const __m256i high = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(bits, 1)), 16);
        // Byte copies accept unaligned checkpoints and create destination float
        // objects in raw allocations without typed-pointer lifetime assumptions.
        std::memcpy(dst + index * sizeof(float), &low, sizeof(low));
        std::memcpy(dst + (index + 8) * sizeof(float), &high, sizeof(high));
    }

    if (index < count) {
        ConvertBf16ToFp32Scalar(src + index * sizeof(uint16_t),
                                dst + index * sizeof(float), count - index);
    }
}

void ConvertFp32ToBf16Avx2(const std::byte* src, std::byte* dst, size_t count) noexcept {
    const __m256i sign_mask = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256i exponent_all_ones = _mm256_set1_epi32(0x7F800000);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i bias_base = _mm256_set1_epi32(0x7FFF);
    size_t index = 0;
    for (; count - index >= 8; index += 8) {
        __m256i bits;
        std::memcpy(&bits, src + index * sizeof(float), sizeof(bits));

        // Both comparison operands have the sign bit cleared, so the signed
        // compare acts as an unsigned magnitude compare: NaN iff magnitude
        // exceeds the all-ones-exponent pattern.
        const __m256i is_nan = _mm256_cmpgt_epi32(
                _mm256_and_si256(bits, sign_mask), exponent_all_ones);
        // NaN lanes preserve sign and the top 7 payload bits, forcing the
        // quiet bit — bitwise identical to the scalar NaN branch.
        const __m256i nan = _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x0040));
        // RNE bias: ties round to even via the LSB of the surviving high half.
        // The addition covers subnormal, normal and overflow boundaries, and
        // keeps infinity encoded as 0x7F80.
        const __m256i bias = _mm256_add_epi32(
                bias_base, _mm256_and_si256(_mm256_srli_epi32(bits, 16), one));
        const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
        const __m256i result = _mm256_blendv_epi8(rounded, nan, is_nan);

        // packus works independently inside 128-bit lanes. Reorder qwords so
        // the low half contains elements 0..7 in order.
        const __m256i packed = _mm256_packus_epi32(result, _mm256_setzero_si256());
        const __m128i output = _mm256_castsi256_si128(_mm256_permute4x64_epi64(packed, 0xD8));
        std::memcpy(dst + index * sizeof(uint16_t), &output, sizeof(output));
    }

    if (index < count) {
        ConvertFp32ToBf16Scalar(src + index * sizeof(float),
                                dst + index * sizeof(uint16_t), count - index);
    }
}

} // namespace aethermind::cpu
