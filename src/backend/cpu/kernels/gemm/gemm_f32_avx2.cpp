#include "../common/dot_product_internal.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_internal.h"

#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__)

constexpr int64_t kAvx2RowsPerPair = 2;
constexpr int64_t kAvx2ColBlock = 4;

// Two rows share every weight stream: per 8-wide K step the two lhs vectors
// stay in registers while the four column weights stream through
// memory-operand FMAs. Peak register usage is 8 accumulators + 2 lhs = 10
// YMM, and the 8 independent accumulator chains saturate both FMA ports.
void RunRowPairAvx2(const float* lhs0, const float* lhs1, const float* rhs,
                    int64_t k, int64_t n, int64_t rhs_n_stride,
                    float* out0, float* out1) noexcept {
    int64_t col = 0;
    for (; col + kAvx2ColBlock <= n; col += kAvx2ColBlock) {
        const float* r0 = rhs + col * rhs_n_stride;
        const float* r1 = r0 + rhs_n_stride;
        const float* r2 = r1 + rhs_n_stride;
        const float* r3 = r2 + rhs_n_stride;

        __m256 s00 = _mm256_setzero_ps();
        __m256 s01 = _mm256_setzero_ps();
        __m256 s02 = _mm256_setzero_ps();
        __m256 s03 = _mm256_setzero_ps();
        __m256 s10 = _mm256_setzero_ps();
        __m256 s11 = _mm256_setzero_ps();
        __m256 s12 = _mm256_setzero_ps();
        __m256 s13 = _mm256_setzero_ps();

        int64_t inner = 0;
        for (; inner + 8 <= k; inner += 8) {
            const __m256 x0 = _mm256_loadu_ps(lhs0 + inner);
            const __m256 x1 = _mm256_loadu_ps(lhs1 + inner);
            s00 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(r0 + inner), s00);
            s01 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(r1 + inner), s01);
            s02 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(r2 + inner), s02);
            s03 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(r3 + inner), s03);
            s10 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(r0 + inner), s10);
            s11 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(r1 + inner), s11);
            s12 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(r2 + inner), s12);
            s13 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(r3 + inner), s13);
        }

        float o00 = HorizontalSumAvx2(s00);
        float o01 = HorizontalSumAvx2(s01);
        float o02 = HorizontalSumAvx2(s02);
        float o03 = HorizontalSumAvx2(s03);
        float o10 = HorizontalSumAvx2(s10);
        float o11 = HorizontalSumAvx2(s11);
        float o12 = HorizontalSumAvx2(s12);
        float o13 = HorizontalSumAvx2(s13);
        for (; inner < k; ++inner) {
            const float x0 = lhs0[inner];
            const float x1 = lhs1[inner];
            o00 += x0 * r0[inner];
            o01 += x0 * r1[inner];
            o02 += x0 * r2[inner];
            o03 += x0 * r3[inner];
            o10 += x1 * r0[inner];
            o11 += x1 * r1[inner];
            o12 += x1 * r2[inner];
            o13 += x1 * r3[inner];
        }

        out0[col] = o00;
        out0[col + 1] = o01;
        out0[col + 2] = o02;
        out0[col + 3] = o03;
        out1[col] = o10;
        out1[col + 1] = o11;
        out1[col + 2] = o12;
        out1[col + 3] = o13;
    }

    for (; col < n; ++col) {
        out0[col] = DotProductF32Avx2(lhs0, rhs + col * rhs_n_stride, static_cast<std::size_t>(k));
        out1[col] = DotProductF32Avx2(lhs1, rhs + col * rhs_n_stride, static_cast<std::size_t>(k));
    }
}

#endif

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGemmF32Avx2Fma(const GemmF32Args& args) noexcept {
    if (args.m >= 1 && args.m <= 8 && args.k > 0 && args.n >= 0 &&
        args.lhs_k_stride == 1 && args.rhs_k_stride == 1 &&
        args.output_n_stride == 1) {
        // Small-M K-contiguous fast path: row pairs share each weight stream
        // (2 rows x 4 columns x 8-wide K), a leftover odd row reuses the
        // single-row multi-target core. k and n are validated above, so the
        // signed-to-unsigned casts are safe. Note: K-panel cache blocking was
        // measured and removed — per-panel reduction/accumulator rebuild cost
        // outweighed the cache reuse win (see benchmark history for details).
        int64_t row = 0;
        for (; row + kAvx2RowsPerPair <= args.m; row += kAvx2RowsPerPair) {
            RunRowPairAvx2(args.lhs + row * args.lhs_m_stride,
                           args.lhs + (row + 1) * args.lhs_m_stride,
                           args.rhs, args.k, args.n, args.rhs_n_stride,
                           args.output + row * args.output_m_stride,
                           args.output + (row + 1) * args.output_m_stride);
        }

        if (row < args.m) {
            DotProductF32MultiTargetAvx2(
                    args.lhs + row * args.lhs_m_stride, args.rhs,
                    static_cast<std::size_t>(args.k), static_cast<std::size_t>(args.n),
                    static_cast<std::size_t>(args.rhs_n_stride),
                    args.output + row * args.output_m_stride);
        }
        return Status::Ok();
    }
    return RunGemmF32Scalar(args);
}
#endif

} // namespace aethermind::cpu::detail