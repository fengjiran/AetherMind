#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_a16_w16_internal.h"

#if defined(__AVX512F__) && defined(__AVX512FP16__)
#include <immintrin.h>
#endif

// AVX512-FP16 A16/W16 GEMM. FP16 elements are widened with the native
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
Status RunGemmAF16WF16Avx512Fp16(const GemmA16W16Args& params) noexcept {
    const GemmArgs& args = params.gemm;
    if (params.operand_dtype != DataType::Float(16) ||
        args.lhs_k_stride != 1 || args.rhs_k_stride != 1 ||
        args.m == 0 || args.n == 0 || args.k == 0) {
        return RunGemmA16W16Reference(params);
    }

    const size_t output_bytes = A16ElementBytes(params.output_dtype);
    for (int64_t row = 0; row < args.m; ++row) {
        const std::byte* const input_row =
                args.lhs + static_cast<size_t>(row * args.lhs_m_stride) * sizeof(uint16_t);
        for (int64_t col = 0; col < args.n; ++col) {
            const std::byte* const weight_row =
                    args.rhs +
                    static_cast<size_t>(col * args.rhs_n_stride) * sizeof(uint16_t);
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
