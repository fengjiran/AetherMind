#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "linear_internal.h"

#if defined(__AVX512F__) && defined(__AVX512FP16__)
#include <immintrin.h>
#endif

#include <array>

// AVX512-FP16 native A16 Linear. FP16 elements are widened with the native
// conversion and accumulated with fused multiply-add over 16 F32 lanes. FP16
// products are exact in F32, so only the fused accumulation and lane reduction
// order differ from the scalar reference; this TU owns that contract and is
// isolated by ISA-specific compile flags.

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

float DotProduct(const std::byte* input, const std::byte* weight, int64_t in_features) noexcept {
    __m512 accumulator = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + 16 <= in_features; k += 16) {
        accumulator = _mm512_fmadd_ps(LoadF16x16(input + k * 2),
                                      LoadF16x16(weight + k * 2),
                                      accumulator);
    }

    float sum = ReduceLanes(accumulator);
    for (; k < in_features; ++k) {
        const float product = LoadA16Element(input + k * 2, DataType::Float(16)) *
                              LoadA16Element(weight + k * 2, DataType::Float(16));
        sum = sum + product;
    }
    return sum;
}

#endif

} // namespace

#if defined(__AVX512F__) && defined(__AVX512FP16__)
Status RunLinearA16F16Avx512Fp16(const LinearA16KernelArgs& args) noexcept {
    if (args.act_dtype != DataType::Float(16) ||
        args.input_col_stride != 1 || args.weight_col_stride != 1 ||
        args.row_count == 0 || args.out_features == 0 || args.in_features == 0) {
        return RunLinearA16Reference(args);
    }

    const size_t output_bytes = A16ElementBytes(args.output_dtype);
    for (int64_t row = 0; row < args.row_count; ++row) {
        const std::byte* const input_row =
                args.input + static_cast<size_t>(row * args.input_row_stride) * sizeof(uint16_t);
        for (int64_t col = 0; col < args.out_features; ++col) {
            const std::byte* const weight_row =
                    args.weight +
                    static_cast<size_t>(col * args.weight_row_stride) * sizeof(uint16_t);
            StoreA16Element(
                    args.output +
                            static_cast<size_t>(row * args.output_row_stride +
                                                col * args.output_col_stride) *
                                    output_bytes,
                    DotProduct(input_row, weight_row, args.in_features),
                    args.output_dtype);
        }
    }
    return Status::Ok();
}
#endif

} // namespace aethermind::cpu::detail
