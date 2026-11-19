#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "aethermind/dtypes/bf16_conversion.h"
#include "rmsnorm_internal.h"

#if defined(__AVX512F__) && defined(__AVX512BF16__)
#include <immintrin.h>
#endif

#include <array>
#include <cmath>

// AVX512-BF16 native RMSNorm. BF16 widens with an exact left shift, the
// variance reduces in 16 F32 lanes, and output narrowing shares the scalar bit
// contract. This TU is compiled with -mavx512f -mavx512bf16 in isolation.

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX512F__) && defined(__AVX512BF16__)
AM_ALWAYS_INLINE __m512 LoadBf16x16(const std::byte* data) noexcept {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

#endif

} // namespace

#if defined(__AVX512F__) && defined(__AVX512BF16__)
Status RunRmsNormA16Bf16Avx512Bf16(const RmsNormA16W32KernelArgs& args) noexcept {
    if (args.act_dtype != DataType::BFloat(16) ||
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
            const __m512 value = LoadBf16x16(input_row + index * 2);
            accumulator = _mm512_fmadd_ps(value, value, accumulator);
        }

        float sum_sq = ReduceLanes(accumulator);
        for (; index < args.hidden_size; ++index) {
            const float value = LoadA16Element(input_row + index * 2, DataType::BFloat(16));
            sum_sq = sum_sq + value * value;
        }

        const float inv_rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(args.hidden_size) +
                                               args.eps);
        const __m512 inv_rms_vector = _mm512_set1_ps(inv_rms);
        index = 0;
        alignas(64) std::array<float, 16> lanes{};
        for (; index + 16 <= args.hidden_size; index += 16) {
            const __m512 value = LoadBf16x16(input_row + index * 2);
            const __m512 weight = _mm512_loadu_ps(args.weight + index);
            const __m512 scaled = _mm512_mul_ps(_mm512_mul_ps(value, inv_rms_vector), weight);
            // VCVTNEPS2BF16 flushes F32 subnormals regardless of MXCSR. Use the
            // same RNE/gradual-underflow narrowing as the scalar tail instead.
            _mm512_store_ps(lanes.data(), scaled);
            ConvertFp32ToBf16Scalar(reinterpret_cast<const std::byte*>(lanes.data()),
                                    output_row + index * 2, lanes.size());
        }

        for (; index < args.hidden_size; ++index) {
            const float value = LoadA16Element(input_row + index * 2, DataType::BFloat(16));
            StoreA16Element(output_row + index * 2,
                            value * inv_rms * args.weight[index], DataType::BFloat(16));
        }
    }
    return Status::Ok();
}
#endif

} // namespace aethermind::cpu::detail
