#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "rmsnorm_internal.h"

#include <cmath>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#endif

// AVX2/FMA widening kernel for A16 RMSNorm. Activations are widened to F32
// (F16C for FP16, an exact left shift for BF16), the variance is reduced in
// eight F32 lanes, and each output element is narrowed once with the shared
// scalar narrowing contract. This TU is compiled with -mavx2 -mfma -mf16c.

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
AM_ALWAYS_INLINE __m256 LoadEightF16(const std::byte* data) noexcept {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data)));
}

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
Status RunAvx2(const RmsNormA16KernelArgs& args) noexcept {
    const DataType dtype = Bfloat ? DataType::BFloat(16) : DataType::Float(16);
    if (args.input_col_stride != 1 || args.weight_stride != 1 ||
        args.output_col_stride != 1 || args.row_count == 0 || args.hidden_size <= 0) {
        return RunRmsNormA16Reference(args);
    }

    for (int64_t row = 0; row < args.row_count; ++row) {
        const std::byte* const input_row =
                args.input + static_cast<size_t>(row * args.input_row_stride) * sizeof(uint16_t);
        std::byte* const output_row =
                args.output + static_cast<size_t>(row * args.output_row_stride) * sizeof(uint16_t);

        __m256 accumulator = _mm256_setzero_ps();
        int64_t index = 0;
        for (; index + 8 <= args.hidden_size; index += 8) {
            const __m256 value = LoadEight<Bfloat>(input_row + index * 2);
            accumulator = _mm256_fmadd_ps(value, value, accumulator);
        }

        float sum_sq = HorizontalSumAvx2(accumulator);
        for (; index < args.hidden_size; ++index) {
            const float value = LoadA16Element(input_row + index * 2, dtype);
            sum_sq = sum_sq + value * value;
        }

        const float inv_rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(args.hidden_size) +
                                               args.eps);
        const __m256 inv_rms_vector = _mm256_set1_ps(inv_rms);
        index = 0;
        alignas(32) float lanes[8] = {};
        for (; index + 8 <= args.hidden_size; index += 8) {
            const __m256 value = LoadEight<Bfloat>(input_row + index * 2);
            const __m256 weight = _mm256_loadu_ps(args.weight + index);
            _mm256_store_ps(lanes, _mm256_mul_ps(_mm256_mul_ps(value, inv_rms_vector), weight));
            for (int lane = 0; lane < 8; ++lane) {
                StoreA16Element(output_row + (index + lane) * 2, lanes[lane], dtype);
            }
        }

        for (; index < args.hidden_size; ++index) {
            const float value = LoadA16Element(input_row + index * 2, dtype);
            StoreA16Element(output_row + index * 2,
                            value * inv_rms * args.weight[index], dtype);
        }
    }
    return Status::Ok();
}

#endif

} // namespace

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
Status RunRmsNormA16Avx2Fma(const RmsNormA16KernelArgs& args) noexcept {
    if (args.act_dtype == DataType::BFloat(16)) {
        return RunAvx2<true>(args);
    }
    return RunAvx2<false>(args);
}
#endif

} // namespace aethermind::cpu::detail
