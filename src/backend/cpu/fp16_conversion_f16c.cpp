#include "aethermind/dtypes/fp16_conversion.h"
#include "fp16_conversion_internal.h"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace aethermind::cpu {

void ConvertFp16ToFp32F16c(const std::byte* src, std::byte* dst, size_t count) noexcept {
    const __m128i exponent_mask = _mm_set1_epi16(0x7C00);
    const __m128i fraction_mask = _mm_set1_epi16(0x03FF);
    const __m128i quiet_bit = _mm_set1_epi16(0x0200);
    const __m128i zero = _mm_setzero_si128();

    size_t index = 0;
    for (; count - index >= 8; index += 8) {
        __m128i bits;
        std::memcpy(&bits, src + index * sizeof(uint16_t), sizeof(bits));
        const __m128i special = _mm_cmpeq_epi16(_mm_and_si128(bits, exponent_mask), exponent_mask);
        const __m128i fraction_zero = _mm_cmpeq_epi16(_mm_and_si128(bits, fraction_mask), zero);
        const __m128i nan = _mm_andnot_si128(fraction_zero, special);
        // Quiet in the integer domain before F16C can observe a signaling NaN.
        bits = _mm_or_si128(bits, _mm_and_si128(nan, quiet_bit));
        const __m256 values = _mm256_cvtph_ps(bits);
        // Byte copies accept unaligned checkpoints and create destination float
        // objects in raw allocations without typed-pointer lifetime assumptions.
        std::memcpy(dst + index * sizeof(float), &values, sizeof(values));
    }

    if (index < count) {
        ConvertFp16ToFp32Scalar(src + index * sizeof(uint16_t),
                                dst + index * sizeof(float), count - index);
    }
}

} // namespace aethermind::cpu
