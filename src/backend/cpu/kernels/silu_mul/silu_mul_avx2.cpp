#include "silu_mul_internal.h"

#include <bit>
#include <cstring>
#include <immintrin.h>

namespace aethermind::cpu::detail {
namespace {

// For x in [-16, 0], reduce x = n*ln(2) + r with |r| <= ln(2)/2.
// The degree-seven Taylor polynomial has absolute truncation error < 8e-9
// on this interval for r, before FP32 rounding. 2^n is always normal here.
__m256 ExpNegative(__m256 x) noexcept {
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0x1.715476p+0F)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 high = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1F), x);
    const __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20F), high);
    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(1.0F / 5040.0F), r, _mm256_set1_ps(1.0F / 720.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 120.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 24.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 6.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    const __m256i bias = _mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127));
    return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(bias, 23)));
}

__m256i AbsBits(__m256 x) noexcept {
    return _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x7fff'ffff));
}

__m256i OutsideSafeMagnitudes(__m256i magnitude, float max) noexcept {
    const __m256i too_large = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(std::bit_cast<int>(max)));
    const __m256i too_small = _mm256_cmpgt_epi32(_mm256_set1_epi32(std::bit_cast<int>(0x1p-32F)), magnitude);
    const __m256i zero = _mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256());
    return _mm256_or_si256(too_large, _mm256_andnot_si256(zero, too_small));
}

bool TrySiluMulEight(__m256 gate, __m256 up, __m256& product) noexcept {
    const __m256i magnitude = AbsBits(gate);
    // Within these bounds, nonzero intermediates and products stay normal.
    const auto outside = _mm256_or_si256(OutsideSafeMagnitudes(magnitude, 16.0F),
                                         OutsideSafeMagnitudes(AbsBits(up), 0x1p100F));
    if (_mm256_movemask_epi8(outside) != 0) {
        return false;
    }
    const __m256 negative_abs = _mm256_castsi256_ps(_mm256_or_si256(magnitude, _mm256_set1_epi32(INT32_MIN)));
    const __m256 e = ExpNegative(negative_abs);
    const __m256 numerator = _mm256_blendv_ps(gate, _mm256_mul_ps(gate, e), gate);
    const __m256 silu = _mm256_div_ps(numerator, _mm256_add_ps(_mm256_set1_ps(1.0F), e));
    product = _mm256_mul_ps(silu, up);
    return true;
}

__m256 LoadBf16Eight(const std::byte* source) noexcept {
    __m128i bits;
    std::memcpy(&bits, source, sizeof(bits));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
}

bool TryStoreBf16Eight(std::byte* destination, __m256 product) noexcept {
    const __m256i bits = _mm256_castps_si256(product);
    const auto low_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0xffff));
    const auto distance = _mm256_abs_epi32(_mm256_sub_epi32(low_bits, _mm256_set1_epi32(0x8000)));
    // Allow margin for the FP32 approximation before deciding which side of
    // the BF16 midpoint to round to. Ambiguous blocks use the scalar formula.
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(_mm256_set1_epi32(33), distance)) != 0) {
        return false;
    }
    const auto lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const auto bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff));
    const auto narrowed = _mm256_srli_epi32(_mm256_add_epi32(bits, bias), 16);
    const auto packed = _mm_packus_epi32(_mm256_castsi256_si128(narrowed), _mm256_extracti128_si256(narrowed, 1));
    std::memcpy(destination, &packed, sizeof(packed));
    return true;
}

} // namespace

Status RunSiluMulF32Avx2Fma(const SiluMulA32KernelArgs& args) noexcept {
    if (_MM_GET_ROUNDING_MODE() != _MM_ROUND_NEAREST) {
        return RunSiluMulF32Reference(args);
    }

    int64_t i = 0;
    for (; i <= args.numel - 8; i += 8) {
        const __m256 gate = _mm256_loadu_ps(args.gate_data + i);
        const __m256 up = _mm256_loadu_ps(args.up_data + i);
        __m256 product;
        if (TrySiluMulEight(gate, up, product)) {
            _mm256_storeu_ps(args.output_data + i, product);
        } else {
            RunSiluMulF32FlatReference(args.gate_data + i, args.up_data + i, args.output_data + i, 8);
        }
    }
    if (i < args.numel) {
        RunSiluMulF32FlatReference(args.gate_data + i, args.up_data + i, args.output_data + i, args.numel - i);
    }
    return Status::Ok();
}

Status RunSiluMulBf16Avx2Fma(const SiluMulA16KernelArgs& args) noexcept {
    if (_MM_GET_ROUNDING_MODE() != _MM_ROUND_NEAREST) {
        return RunSiluMulA16Reference(args);
    }

    const auto scalar = [&](int64_t i, int64_t length) {
        const size_t offset = static_cast<size_t>(i) * sizeof(uint16_t);
        RunSiluMulA16FlatReference(args.gate_data + offset, args.up_data + offset,
                                   args.output_data + offset, args.act_dtype, length);
    };
    int64_t i = 0;
    for (; i <= args.numel - 8; i += 8) {
        const size_t offset = static_cast<size_t>(i) * sizeof(uint16_t);
        const __m256 gate = LoadBf16Eight(args.gate_data + offset);
        const __m256 up = LoadBf16Eight(args.up_data + offset);
        __m256 product;
        if (!TrySiluMulEight(gate, up, product) || !TryStoreBf16Eight(args.output_data + offset, product)) {
            scalar(i, 8);
        }
    }
    if (i < args.numel) {
        scalar(i, args.numel - i);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
