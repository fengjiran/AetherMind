#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_a16_w16_internal.h"

#if defined(__AVX512F__) && defined(__AVX512BF16__)
#include <immintrin.h>
#endif

// AVX512-BF16 A16/W16 GEMM. VDPBF16PS accumulates the pairwise products of
// 32 BF16 elements into 16 F32 lanes, but always applies DAZ/FTZ. Only a proven
// safe magnitude domain uses it; other operands execute through the reference.
// This TU owns the grouped summation order and has isolated ISA flags.

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX512F__) && defined(__AVX512BF16__)
AM_ALWAYS_INLINE __m512bh LoadBf16x32(const std::byte* data) noexcept {
    return __m512bh(_mm512_loadu_si512(data));
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
Status RunGemmABf16WBf16Avx512Bf16(const GemmA16W16Args& params) noexcept {
    const GemmArgs& args = params.gemm;
    if (params.operand_dtype != DataType::BFloat(16) ||
        args.lhs_k_stride != 1 || args.rhs_k_stride != 1 ||
        args.m == 0 || args.n == 0 || args.k == 0) {
        return RunGemmA16W16Reference(params);
    }

    const size_t output_bytes = A16ElementBytes(params.output_dtype);
    const int64_t vector_elements = args.k - args.k % 32;
    for (int64_t row = 0; row < args.m; ++row) {
        const std::byte* const input_row =
                args.lhs + static_cast<size_t>(row * args.lhs_m_stride) * sizeof(uint16_t);
        for (int64_t col = 0; col < args.n; ++col) {
            const std::byte* const weight_row =
                    args.rhs +
                    static_cast<size_t>(col * args.rhs_n_stride) * sizeof(uint16_t);
            if (!IsBf16NativeDotProductSafe(input_row, weight_row, vector_elements)) {
                // Input/weight are disjoint from output. Recomputing the whole
                // invocation safely replaces any outputs already produced.
                return RunGemmA16W16Reference(params);
            }
            StoreA16Element(
                    args.output +
                            static_cast<size_t>(row * args.output_m_stride +
                                                col * args.output_n_stride) *
                                    output_bytes,
                    DotProduct(input_row, weight_row, args.k),
                    params.output_dtype);
        }
    }
    return Status::Ok();
}
#endif

} // namespace aethermind::cpu::detail
