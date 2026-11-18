#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "backend/cpu/kernels/common/dot_product_internal.h"
#include "gemm_internal.h"

#include <algorithm>
#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include "gemm_avx2_internal.h"
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
// accumulators plus 2 streaming B vectors, and the 8 FMAs issued per K step
// cover 64 output lanes to saturate both FMA ports.

constexpr int64_t kBlockedMR = kGemmAvx2MR;
constexpr int64_t kBlockedNR = kGemmAvx2NR;
constexpr int64_t kBlockedKC = kGemmAvx2KC;

// Transposed k-major layout: buf_b[kk * 16 + col] = B[(nb_base + col)][kb + kk].
void PackBPanel(const float* rhs, int64_t rhs_n_stride, int64_t kb,
                int64_t nb_base, int64_t kc_len, float* buf_b) noexcept {
    const float* source = rhs + nb_base * rhs_n_stride + kb;
    const int64_t vector_end = kc_len - kc_len % 8;
    for (int64_t col = 0; col < kBlockedNR; col += 8) {
        for (int64_t kk = 0; kk < vector_end; kk += 8) {
            __m256 rows[8];
            for (int64_t lane = 0; lane < 8; ++lane) {
                rows[lane] = _mm256_loadu_ps(source + (col + lane) * rhs_n_stride + kk);
            }
            StoreGemmBTile8x8Avx2(rows, buf_b + kk * kBlockedNR + col);
        }
    }

    for (int64_t kk = vector_end; kk < kc_len; ++kk) {
        for (int64_t col = 0; col < kBlockedNR; ++col) {
            buf_b[kk * kBlockedNR + col] = source[col * rhs_n_stride + kk];
        }
    }
}

// out rows use unit output_n_stride (validated by the caller);
// out_row points at output[mb_base][nb_base].
void RunBlockedGemmAvx2(const GemmArgs& args, int64_t m_full, int64_t n_full) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const float*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);

    alignas(32) float buf_b[static_cast<size_t>(kBlockedKC * kBlockedNR)];

    // Unit K stride permits direct A broadcasts and lets all row tiles
    // consume a B panel before the next panel is packed.
    for (int64_t kb = 0; kb < args.k; kb += kBlockedKC) {
        const int64_t kc_len = std::min(kBlockedKC, args.k - kb);
        for (int64_t nb = 0; nb < n_full; nb += kBlockedNR) {
            PackBPanel(rhs_data, args.rhs_n_stride, kb, nb, kc_len, buf_b);
            for (int64_t row = 0; row < m_full; row += kBlockedMR) {
                AccumulateF32Tile4x16Avx2(lhs_data + row * args.lhs_m_stride + kb, buf_b, kc_len,
                                          args.output_m_stride, output_data + row * args.output_m_stride + nb,
                                          args.lhs_m_stride);
            }
        }
    }
}

#endif

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGemmF32Avx2Fma(const GemmArgs& args) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const float*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);

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
            RunRowPairAvx2(lhs_data + row * args.lhs_m_stride,
                           lhs_data + (row + 1) * args.lhs_m_stride,
                           rhs_data, args.k, args.n, args.rhs_n_stride,
                           output_data + row * args.output_m_stride,
                           output_data + (row + 1) * args.output_m_stride);
        }

        if (row < args.m) {
            DotProductF32MultiTargetAvx2(
                    lhs_data + row * args.lhs_m_stride, rhs_data,
                    static_cast<std::size_t>(args.k), static_cast<std::size_t>(args.n),
                    static_cast<std::size_t>(args.rhs_n_stride),
                    output_data + row * args.output_m_stride);
        }
        return Status::Ok();
    }

    if (args.m >= 9 && args.k > 0 && args.n >= 0 &&
        args.lhs_k_stride == 1 && args.rhs_k_stride == 1 &&
        args.output_n_stride == 1) {
        // Blocked prefill path: the 4x16 micro-kernel covers the aligned M x N
        // core (zeroed first, then accumulated across K panels); the right
        // N-tail columns and the bottom M-tail rows reuse the row-pair and
        // single-row cores with overwrite semantics.
        const int64_t m_full = args.m - args.m % kBlockedMR;
        const int64_t n_full = args.n - args.n % kBlockedNR;
        for (int64_t row = 0; row < m_full; ++row) {
            float* out_row = output_data + row * args.output_m_stride;
            for (int64_t col = 0; col < n_full; ++col) {
                out_row[col] = 0.0f;
            }
        }

        RunBlockedGemmAvx2(args, m_full, n_full);
        if (const int64_t n_tail = args.n - n_full; n_tail > 0) {
            int64_t row = 0;
            for (; row + kAvx2RowsPerPair <= m_full; row += kAvx2RowsPerPair) {
                RunRowPairAvx2(lhs_data + row * args.lhs_m_stride,
                               lhs_data + (row + 1) * args.lhs_m_stride,
                               rhs_data + n_full * args.rhs_n_stride,
                               args.k, n_tail, args.rhs_n_stride,
                               output_data + row * args.output_m_stride + n_full,
                               output_data + (row + 1) * args.output_m_stride + n_full);
            }

            if (row < m_full) {
                DotProductF32MultiTargetAvx2(
                        lhs_data + row * args.lhs_m_stride,
                        rhs_data + n_full * args.rhs_n_stride,
                        static_cast<std::size_t>(args.k),
                        static_cast<std::size_t>(n_tail),
                        static_cast<std::size_t>(args.rhs_n_stride),
                        output_data + row * args.output_m_stride + n_full);
            }
        }

        int64_t row = m_full;
        for (; row + kAvx2RowsPerPair <= args.m; row += kAvx2RowsPerPair) {
            RunRowPairAvx2(lhs_data + row * args.lhs_m_stride,
                           lhs_data + (row + 1) * args.lhs_m_stride,
                           rhs_data, args.k, args.n, args.rhs_n_stride,
                           output_data + row * args.output_m_stride,
                           output_data + (row + 1) * args.output_m_stride);
        }

        if (row < args.m) {
            DotProductF32MultiTargetAvx2(
                    lhs_data + row * args.lhs_m_stride, rhs_data,
                    static_cast<std::size_t>(args.k), static_cast<std::size_t>(args.n),
                    static_cast<std::size_t>(args.rhs_n_stride),
                    output_data + row * args.output_m_stride);
        }
        return Status::Ok();
    }
    return RunGemmF32Scalar(args);
}
#endif

} // namespace aethermind::cpu::detail
