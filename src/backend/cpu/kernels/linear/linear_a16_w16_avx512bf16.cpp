#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "linear_internal.h"

#if defined(__AVX512F__) && defined(__AVX512BF16__)
#include <immintrin.h>
#endif

#include <array>

// AVX512-BF16 native A16 Linear. VDPBF16PS accumulates the pairwise products of
// 32 BF16 elements into 16 F32 lanes, but always applies DAZ/FTZ. Only a proven
// safe magnitude domain uses it; other operands execute through the reference.
// This TU owns the grouped summation order and has isolated ISA flags.

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX512F__) && defined(__AVX512BF16__)
AM_ALWAYS_INLINE __m512bh LoadBf16x32(const std::byte* data) noexcept {
    return __m512bh(_mm512_loadu_si512(data));
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
    for (; k + 32 <= in_features; k += 32) {
        accumulator = _mm512_dpbf16_ps(accumulator,
                                       LoadBf16x32(input + k * 2),
                                       LoadBf16x32(weight + k * 2));
    }

    float sum = ReduceLanes(accumulator);
    for (; k < in_features; ++k) {
        const float product = LoadA16Element(input + k * 2, DataType::BFloat(16)) *
                              LoadA16Element(weight + k * 2, DataType::BFloat(16));
        sum = sum + product;
    }
    return sum;
}

#endif

} // namespace

#if defined(__AVX512F__) && defined(__AVX512BF16__)
Status RunLinearA16Bf16Avx512Bf16(const LinearA16W16KernelArgs& args) noexcept {
    if (args.act_dtype != DataType::BFloat(16) ||
        args.input_col_stride != 1 || args.weight_col_stride != 1 ||
        args.row_count == 0 || args.out_features == 0 || args.in_features == 0) {
        return RunLinearA16Reference(args);
    }

    const size_t output_bytes = A16ElementBytes(args.output_dtype);
    const int64_t vector_elements = args.in_features - args.in_features % 32;
    for (int64_t row = 0; row < args.row_count; ++row) {
        const std::byte* const input_row =
                args.input + static_cast<size_t>(row * args.input_row_stride) * sizeof(uint16_t);
        for (int64_t col = 0; col < args.out_features; ++col) {
            const std::byte* const weight_row =
                    args.weight +
                    static_cast<size_t>(col * args.weight_row_stride) * sizeof(uint16_t);
            if (!IsBf16NativeDotProductSafe(input_row, weight_row, vector_elements)) {
                // Input/weight are disjoint from output. Recomputing the whole
                // invocation safely replaces any outputs already produced.
                return RunLinearA16Reference(args);
            }
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
