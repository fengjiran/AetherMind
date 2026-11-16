#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "rmsnorm_internal.h"

#if defined(__AVX512F__) && defined(__AVX512FP16__)
#include <immintrin.h>
#endif

#include <array>
#include <cmath>

// AVX512-FP16 native RMSNorm. FP16 elements are widened and narrowed with the
// native conversions and the variance reduces in 16 F32 lanes. This TU is
// compiled with -mavx512f -mavx512fp16 in isolation.

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX512F__) && defined(__AVX512FP16__)
AM_ALWAYS_INLINE __m512 LoadF16x16(const std::byte* data) noexcept {
    return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data)));
}

AM_ALWAYS_INLINE float ReduceLanes(__m512 value) noexcept {
    alignas(64) std::array<float, 16> lanes{};
    _mm512_store_ps(lanes.data(), value);
    float sum = 0.0F;
    for (const float lane: lanes) {
        sum = sum + lane;
    }
    return sum;
}

#endif

} // namespace

#if defined(__AVX512F__) && defined(__AVX512FP16__)
Status RunRmsNormA16F16Avx512Fp16(const RmsNormA16W32KernelArgs& args) noexcept {
    if (args.act_dtype != DataType::Float(16) ||
        args.input_col_stride != 1 || args.weight_stride != 1 ||
        args.output_col_stride != 1 || args.row_count == 0 || args.hidden_size <= 0) {
        return RunRmsNormA16Reference(args);
    }

    for (int64_t row = 0; row < args.row_count; ++row) {
        const std::byte* const input_row =
                args.input + static_cast<size_t>(row * args.input_row_stride) * sizeof(uint16_t);
        std::byte* const output_row =
                args.output + static_cast<size_t>(row * args.output_row_stride) * sizeof(uint16_t);

        __m512 accumulator = _mm512_setzero_ps();
        int64_t index = 0;
        for (; index + 16 <= args.hidden_size; index += 16) {
            const __m512 value = LoadF16x16(input_row + index * 2);
            accumulator = _mm512_fmadd_ps(value, value, accumulator);
        }

        float sum_sq = ReduceLanes(accumulator);
        for (; index < args.hidden_size; ++index) {
            const float value = LoadA16Element(input_row + index * 2, DataType::Float(16));
            sum_sq = sum_sq + value * value;
        }

        const float inv_rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(args.hidden_size) +
                                               args.eps);
        const __m512 inv_rms_vector = _mm512_set1_ps(inv_rms);
        index = 0;
        for (; index + 16 <= args.hidden_size; index += 16) {
            const __m512 value = LoadF16x16(input_row + index * 2);
            const __m512 weight = _mm512_loadu_ps(args.weight + index);
            const __m512 scaled = _mm512_mul_ps(_mm512_mul_ps(value, inv_rms_vector), weight);
            const __m256i narrowed =
                    _mm512_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_row + index * 2), narrowed);
        }

        for (; index < args.hidden_size; ++index) {
            const float value = LoadA16Element(input_row + index * 2, DataType::Float(16));
            StoreA16Element(output_row + index * 2,
                            value * inv_rms * args.weight[index], DataType::Float(16));
        }
    }
    return Status::Ok();
}
#endif

} // namespace aethermind::cpu::detail
