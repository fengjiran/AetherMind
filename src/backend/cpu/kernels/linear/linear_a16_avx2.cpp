#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "linear_internal.h"

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#endif

// AVX2/FMA widening fallback for A16 Linear. FP16 products are exact in F32
// (11-bit significands) and BF16 products are exact as well (8-bit
// significands), so the only numerical difference from the scalar reference is
// the fused multiply-add (one rounding per accumulate) and the eight-lane
// reduction order. The ELF TU is compiled with -mavx2 -mfma in isolation.

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
AM_ALWAYS_INLINE __m256 LoadEightF16(const std::byte* data) noexcept {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data)));
}

// Left-shifting the 16 BF16 bits into the high half of an F32 lane is exact for
// every finite value, including BF16 subnormals, infinities, and NaNs.
AM_ALWAYS_INLINE __m256 LoadEightBf16(const std::byte* data) noexcept {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

template<bool Bfloat>
AM_ALWAYS_INLINE __m256 LoadEight(const std::byte* data) noexcept {
    if constexpr (Bfloat) {
        return LoadEightBf16(data);
    } else {
        return LoadEightF16(data);
    }
}

template<bool Bfloat>
float DotProduct(const std::byte* input, const std::byte* weight, int64_t in_features) noexcept {
    const DataType dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    __m256 accumulator = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + 8 <= in_features; k += 8) {
        accumulator = _mm256_fmadd_ps(LoadEight<Bfloat>(input + k * 2),
                                      LoadEight<Bfloat>(weight + k * 2),
                                      accumulator);
    }

    float sum = HorizontalSumAvx2(accumulator);
    for (; k < in_features; ++k) {
        const float product = LoadA16Element(input + k * 2, dtype) *
                              LoadA16Element(weight + k * 2, dtype);
        sum = sum + product;
    }
    return sum;
}

template<bool Bfloat>
Status RunAvx2(const LinearA16KernelArgs& args) noexcept {
    // The vector loop requires unit column strides; any other positive layout
    // uses the scalar reference so the prepared arguments never fail.
    if (args.input_col_stride != 1 || args.weight_col_stride != 1 ||
        args.row_count == 0 || args.out_features == 0 || args.in_features == 0) {
        return RunLinearA16Reference(args);
    }

    const size_t input_bytes = A16ElementBytes(args.act_dtype);
    const size_t weight_bytes = A16ElementBytes(args.act_dtype);
    const size_t output_bytes = A16ElementBytes(args.output_dtype);

    for (int64_t row = 0; row < args.row_count; ++row) {
        const std::byte* const input_row =
                args.input + static_cast<size_t>(row * args.input_row_stride) * input_bytes;
        for (int64_t col = 0; col < args.out_features; ++col) {
            const std::byte* const weight_row =
                    args.weight + static_cast<size_t>(col * args.weight_row_stride) * weight_bytes;
            const float sum = DotProduct<Bfloat>(input_row, weight_row, args.in_features);
            StoreA16Element(
                    args.output +
                            static_cast<size_t>(row * args.output_row_stride +
                                                col * args.output_col_stride) *
                                    output_bytes,
                    sum, args.output_dtype);
        }
    }
    return Status::Ok();
}

#endif

} // namespace

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
Status RunLinearA16Avx2Fma(const LinearA16KernelArgs& args) noexcept {
    if (args.act_dtype == DataType::BFloat(16)) {
        return RunAvx2<true>(args);
    }
    return RunAvx2<false>(args);
}
#endif

} // namespace aethermind::cpu::detail
