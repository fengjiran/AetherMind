#include "../common/dot_product_internal.h"
#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_internal.h"

#include <algorithm>
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
        out0[col] = DotProductF32Avx2(lhs0, rhs + col * rhs_n_stride,
                                      static_cast<std::size_t>(k));
        out1[col] = DotProductF32Avx2(lhs1, rhs + col * rhs_n_stride,
                                      static_cast<std::size_t>(k));
    }
}

// ---- Blocked prefill GEMM -------------------------------------------------
//
// 4x16 micro-kernel: each row keeps two 8-wide accumulators spanning the 16
// output columns along N; per scalar K step the row's A value is broadcast
// (set1) and multiplied with the two 8-column vectors of the current B row
// (B is packed transposed, k-major: panel[kk][col]). Peak usage is 8
// accumulators plus 2 streaming B vectors, and the 64 independent FMAs per
// 8-column B slab saturate both FMA ports.

constexpr int64_t kBlockedMR = 4;
constexpr int64_t kBlockedNR = 16;
constexpr int64_t kBlockedKC = 512;
// A rows kept resident per K panel; 48 rows x 512 K x 4B = 96 KiB.
constexpr int64_t kBlockedMB = 48;

void PackAPanel(const float* lhs, int64_t lhs_m_stride, int64_t kb,
                int64_t mb_base, int64_t mb_rows, int64_t kc_len,
                float* buf_a) noexcept {
    for (int64_t r = 0; r < mb_rows; ++r) {
        const float* src = lhs + (mb_base + r) * lhs_m_stride + kb;
        float* dst = buf_a + r * kBlockedKC;
        for (int64_t kk = 0; kk < kc_len; ++kk) {
            dst[kk] = src[kk];
        }
    }
}

// Transposed k-major layout: buf_b[kk * 16 + col] = B[(nb_base + col)][kb + kk].
void PackBPanel(const float* rhs, int64_t rhs_n_stride, int64_t kb,
                int64_t nb_base, int64_t kc_len, float* buf_b) noexcept {
    for (int64_t kk = 0; kk < kc_len; ++kk) {
        for (int64_t col = 0; col < kBlockedNR; ++col) {
            buf_b[kk * kBlockedNR + col] = rhs[(nb_base + col) * rhs_n_stride + kb + kk];
        }
    }
}

// out rows use unit output_n_stride (validated by the caller);
// out_row points at output[mb_base][nb_base].
void MicroKernel4x16(const float* buf_a, const float* buf_b,
                     int64_t kc_len, int64_t output_m_stride,
                     float* out_row0) noexcept {
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps();
    __m256 acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps();
    __m256 acc31 = _mm256_setzero_ps();

    const float* a0 = buf_a;
    const float* a1 = a0 + kBlockedKC;
    const float* a2 = a1 + kBlockedKC;
    const float* a3 = a2 + kBlockedKC;

    for (int64_t kk = 0; kk < kc_len; ++kk) {
        const __m256 x0 = _mm256_set1_ps(a0[kk]);
        const __m256 x1 = _mm256_set1_ps(a1[kk]);
        const __m256 x2 = _mm256_set1_ps(a2[kk]);
        const __m256 x3 = _mm256_set1_ps(a3[kk]);
        const __m256 b0 = _mm256_loadu_ps(buf_b + kk * kBlockedNR);
        const __m256 b1 = _mm256_loadu_ps(buf_b + kk * kBlockedNR + 8);
        acc00 = _mm256_fmadd_ps(x0, b0, acc00);
        acc01 = _mm256_fmadd_ps(x0, b1, acc01);
        acc10 = _mm256_fmadd_ps(x1, b0, acc10);
        acc11 = _mm256_fmadd_ps(x1, b1, acc11);
        acc20 = _mm256_fmadd_ps(x2, b0, acc20);
        acc21 = _mm256_fmadd_ps(x2, b1, acc21);
        acc30 = _mm256_fmadd_ps(x3, b0, acc30);
        acc31 = _mm256_fmadd_ps(x3, b1, acc31);
    }

    float* out0 = out_row0;
    float* out1 = out0 + output_m_stride;
    float* out2 = out1 + output_m_stride;
    float* out3 = out2 + output_m_stride;
    __m256 o00 = _mm256_loadu_ps(out0);
    __m256 o01 = _mm256_loadu_ps(out0 + 8);
    __m256 o10 = _mm256_loadu_ps(out1);
    __m256 o11 = _mm256_loadu_ps(out1 + 8);
    __m256 o20 = _mm256_loadu_ps(out2);
    __m256 o21 = _mm256_loadu_ps(out2 + 8);
    __m256 o30 = _mm256_loadu_ps(out3);
    __m256 o31 = _mm256_loadu_ps(out3 + 8);
    _mm256_storeu_ps(out0, _mm256_add_ps(o00, acc00));
    _mm256_storeu_ps(out0 + 8, _mm256_add_ps(o01, acc01));
    _mm256_storeu_ps(out1, _mm256_add_ps(o10, acc10));
    _mm256_storeu_ps(out1 + 8, _mm256_add_ps(o11, acc11));
    _mm256_storeu_ps(out2, _mm256_add_ps(o20, acc20));
    _mm256_storeu_ps(out2 + 8, _mm256_add_ps(o21, acc21));
    _mm256_storeu_ps(out3, _mm256_add_ps(o30, acc30));
    _mm256_storeu_ps(out3 + 8, _mm256_add_ps(o31, acc31));
}

void RunBlockedGemmAvx2(const GemmF32Args& args, int64_t m_full, int64_t n_full) noexcept {
    alignas(32) float buf_a[static_cast<size_t>(kBlockedMB * kBlockedKC)];
    alignas(32) float buf_b[static_cast<size_t>(kBlockedKC * kBlockedNR)];

    for (int64_t kb = 0; kb < args.k; kb += kBlockedKC) {
        const int64_t kc_len = std::min(kBlockedKC, args.k - kb);
        for (int64_t mb = 0; mb < m_full; mb += kBlockedMB) {
            const int64_t mb_rows = std::min(kBlockedMB, m_full - mb);
            PackAPanel(args.lhs, args.lhs_m_stride, kb, mb, mb_rows, kc_len, buf_a);
            for (int64_t nb = 0; nb < n_full; nb += kBlockedNR) {
                PackBPanel(args.rhs, args.rhs_n_stride, kb, nb, kc_len, buf_b);
                for (int64_t row = 0; row < mb_rows; row += kBlockedMR) {
                    MicroKernel4x16(buf_a + row * kBlockedKC, buf_b, kc_len,
                                    args.output_m_stride,
                                    args.output + (mb + row) * args.output_m_stride + nb);
                }
            }
        }
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

    if (args.m >= 9 && args.k > 0 && args.n >= 0 &&
        args.lhs_k_stride == 1 && args.rhs_k_stride == 1 &&
        args.output_n_stride == 1) {
        // Blocked prefill path: the 6x8 micro-kernel covers the aligned M x N
        // core (zeroed first, then accumulated across K panels); the right
        // N-tail columns and the bottom M-tail rows reuse the row-pair and
        // single-row cores with overwrite semantics.
        const int64_t m_full = args.m - args.m % kBlockedMR;
        const int64_t n_full = args.n - args.n % kBlockedNR;
        for (int64_t row = 0; row < m_full; ++row) {
            float* out_row = args.output + row * args.output_m_stride;
            for (int64_t col = 0; col < n_full; ++col) {
                out_row[col] = 0.0f;
            }
        }

        RunBlockedGemmAvx2(args, m_full, n_full);
        if (const int64_t n_tail = args.n - n_full; n_tail > 0) {
            int64_t row = 0;
            for (; row + kAvx2RowsPerPair <= m_full; row += kAvx2RowsPerPair) {
                RunRowPairAvx2(args.lhs + row * args.lhs_m_stride,
                               args.lhs + (row + 1) * args.lhs_m_stride,
                               args.rhs + n_full * args.rhs_n_stride,
                               args.k, n_tail, args.rhs_n_stride,
                               args.output + row * args.output_m_stride + n_full,
                               args.output + (row + 1) * args.output_m_stride + n_full);
            }

            if (row < m_full) {
                DotProductF32MultiTargetAvx2(
                        args.lhs + row * args.lhs_m_stride,
                        args.rhs + n_full * args.rhs_n_stride,
                        static_cast<std::size_t>(args.k),
                        static_cast<std::size_t>(n_tail),
                        static_cast<std::size_t>(args.rhs_n_stride),
                        args.output + row * args.output_m_stride + n_full);
            }
        }

        int64_t row = m_full;
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