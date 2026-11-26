#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "aethermind/base/macros.h"
#include "backend/cpu/kernels/dot_product/dot_product_internal.h"
#include "gemm_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include "gemm_avx2_internal.h"
#include <immintrin.h>
#endif

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__)

constexpr int64_t kAvx2RowsPerPair = 2;
constexpr int64_t kAvx2ColBlock = 4;

// Two rows share each of four weight vectors across eight independent sums.
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

// Compute a nonempty rectangle directly from K-contiguous weight rows.
void RunKContiguousRowPairsAvx2(const GemmArgs& args, int64_t row_begin, int64_t row_end,
                                int64_t col_begin, int64_t cols) noexcept {
    const auto* lhs = reinterpret_cast<const float*>(args.lhs);
    const auto* rhs = reinterpret_cast<const float*>(args.rhs) + col_begin * args.rhs_n_stride;
    auto* output = reinterpret_cast<float*>(args.output) + col_begin;
    int64_t row = row_begin;
    while (row + kAvx2RowsPerPair <= row_end) {
        RunRowPairAvx2(lhs + row * args.lhs_m_stride,
                       lhs + (row + 1) * args.lhs_m_stride,
                       rhs, args.k, cols, args.rhs_n_stride,
                       output + row * args.output_m_stride,
                       output + (row + 1) * args.output_m_stride);
        row += kAvx2RowsPerPair;
    }

    if (row < row_end) {
        DotProductF32MultiTargetAvx2(
                lhs + row * args.lhs_m_stride, rhs,
                static_cast<size_t>(args.k), static_cast<size_t>(cols),
                static_cast<size_t>(args.rhs_n_stride),
                output + row * args.output_m_stride);
    }
}

// ---- Blocked prefill GEMM -------------------------------------------------
//
// The production 6x16 micro-kernel keeps twelve independent accumulator
// chains. The 6/4/2-row decomposition keeps complete row pairs on the same
// B panel. Tile6DupA is a measurement candidate that
// additionally packs A in broadcast layout (each scalar replicated 8x,
// PackGemmAPanelDupAvx2) so the inner loop runs pure vector loads and FMAs
// with no shuffle-port traffic; it uses KC=256 panels so the duplicated A
// block (48 rows x 256 x 32 B) and the B block (128 cols x 256 x 4 B) stay
// L2-resident and amortize both packs. Measured on Alder Lake it loses to the
// direct-broadcast kernel, so it stays a candidate.

constexpr int64_t kBlockedNR = kGemmAvx2NR;
constexpr int64_t kBlockedKC = kGemmAvx2KC;
constexpr int64_t kTile6MR = kGemmAvx2MR6;
constexpr int64_t kTile6KC = 256;
constexpr int64_t kTile6MC = 48;
constexpr int64_t kTile6NC = 128;

struct F32GemmWeightOps {
    static constexpr size_t kElementBytes = sizeof(float);

    static __m256 LoadEight(const std::byte* data) noexcept {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(data));
    }

    static float LoadOne(const std::byte* data) noexcept {
        float value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
};

// One K step keeps twelve sums, two B vectors, and one A broadcast live.
// Keeping this loop separate from the panel driver lets the sums stay in YMM
// registers across the full panel.
AM_NOINLINE void AccumulateDirectF32Tile6x16Avx2(
        const float* lhs, const float* panel, int64_t depth, int64_t output_stride,
        float* output, int64_t lhs_stride) noexcept {
    struct RowSum {
        __m256 lo = _mm256_setzero_ps();
        __m256 hi = _mm256_setzero_ps();
    };

    RowSum acc[kTile6MR];
    const float* rows[kTile6MR];
    for (int row = 0; row < kTile6MR; ++row) {
        rows[row] = lhs + row * lhs_stride;
    }

    for (int64_t k = 0; k < depth; ++k) {
        const __m256 lo = _mm256_load_ps(panel + k * kBlockedNR);
        const __m256 hi = _mm256_load_ps(panel + k * kBlockedNR + 8);
        for (int row = 0; row < kTile6MR; ++row) {
            const __m256 a = _mm256_broadcast_ss(rows[row] + k);
            acc[row].lo = _mm256_fmadd_ps(a, lo, acc[row].lo);
            acc[row].hi = _mm256_fmadd_ps(a, hi, acc[row].hi);
        }
    }

    for (int row = 0; row < kTile6MR; ++row) {
        float* out = output + row * output_stride;
        _mm256_storeu_ps(out, _mm256_add_ps(_mm256_loadu_ps(out), acc[row].lo));
        _mm256_storeu_ps(out + 8, _mm256_add_ps(_mm256_loadu_ps(out + 8), acc[row].hi));
    }
}

void RunBlockedGemmTile6Avx2(const GemmArgs& args, int64_t m_full, int64_t n_full) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const std::byte*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);
    const size_t rhs_row_bytes = static_cast<size_t>(args.rhs_n_stride) * sizeof(float);
    const int64_t paired_rows = m_full - m_full % 2;

    alignas(32) float buf_b[static_cast<size_t>(kBlockedKC * kBlockedNR)];

    for (int64_t kb = 0; kb < args.k; kb += kBlockedKC) {
        const int64_t kc_len = std::min(kBlockedKC, args.k - kb);
        for (int64_t nb = 0; nb < n_full; nb += kBlockedNR) {
            const auto* const b_source =
                    rhs_data +
                    static_cast<size_t>(nb * args.rhs_n_stride + kb) * sizeof(float);
            PackGemmBPanelAvx2<F32GemmWeightOps, false>(
                    b_source, rhs_row_bytes, kc_len, buf_b);
            for (int64_t row = 0; row < paired_rows;) {
                const int64_t remaining = paired_rows - row;
                if (remaining >= kTile6MR) {
                    AccumulateDirectF32Tile6x16Avx2(
                            lhs_data + row * args.lhs_m_stride + kb,
                            buf_b, kc_len, args.output_m_stride,
                            output_data + row * args.output_m_stride + nb,
                            args.lhs_m_stride);
                    row += kTile6MR;
                } else if (remaining >= 4) {
                    AccumulateF32Tile4x16UnrolledAvx2(
                            lhs_data + row * args.lhs_m_stride + kb,
                            buf_b, kc_len, args.output_m_stride,
                            output_data + row * args.output_m_stride + nb,
                            args.lhs_m_stride);
                    row += 4;
                } else {
                    AccumulateF32Tile2x16UnrolledAvx2(
                            lhs_data + row * args.lhs_m_stride + kb,
                            buf_b, kc_len, args.output_m_stride,
                            output_data + row * args.output_m_stride + nb,
                            args.lhs_m_stride);
                    row += 2;
                }
            }

            if (paired_rows < m_full) {
                const float* lhs_row = lhs_data + paired_rows * args.lhs_m_stride + kb;
                AccumulateF32TileNx16CoreAvx2<1>(
                        buf_b, kc_len, args.output_m_stride,
                        output_data + paired_rows * args.output_m_stride + nb,
                        [=](int, int64_t k) { return _mm256_broadcast_ss(lhs_row + k); });
            }
        }
    }
}

// Packs an A panel in broadcast layout: every scalar is replicated eight
// times so the kernel replaces broadcast shuffles with plain vector loads.
// Row stride is kc_len * 8 floats; the caller keeps the panel aligned.
void PackGemmAPanelDupAvx2(const float* lhs, int64_t lhs_m_stride,
                           int64_t kb, int64_t mb_base, int64_t mb_rows,
                           int64_t kc_len, float* buf_a) noexcept {
    for (int64_t row = 0; row < mb_rows; ++row) {
        const float* const source = lhs + (mb_base + row) * lhs_m_stride + kb;
        float* const destination = buf_a + row * kc_len * 8;
        for (int64_t kk = 0; kk < kc_len; ++kk) {
            _mm256_store_ps(destination + kk * 8, _mm256_broadcast_ss(source + kk));
        }
    }
}

// 6x16 tile reading A from a broadcast-packed panel (see PackGemmAPanelDupAvx2):
// the inner loop issues only aligned vector loads and FMAs, no shuffles.
void AccumulateF32Tile6x16DupAvx2(const float* buf_a, const float* buf_b,
                                  int64_t kc_len, int64_t output_m_stride,
                                  float* out_row0, int64_t a_row_stride) noexcept {
    AccumulateF32TileNx16CoreAvx2<6>(
            buf_b, kc_len, output_m_stride, out_row0,
            [=](int r, int64_t k) {
                return _mm256_load_ps(buf_a + r * a_row_stride + k * 8);
            });
}

void RunBlockedGemmTile6DupAAvx2(const GemmArgs& args, int64_t m_full, int64_t n_full) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const std::byte*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);
    const size_t rhs_row_bytes = static_cast<size_t>(args.rhs_n_stride) * sizeof(float);

    // Both panels are reused across the other operand's block loop, which is
    // what lets the duplicated A layout amortize its 8x size amplification.
    alignas(32) float buf_a[static_cast<size_t>(kTile6MC * kTile6KC * 8)];
    alignas(32) float buf_b[static_cast<size_t>(kTile6NC * kTile6KC)];

    for (int64_t kb = 0; kb < args.k; kb += kTile6KC) {
        const int64_t kc_len = std::min(kTile6KC, args.k - kb);
        for (int64_t mb = 0; mb < m_full; mb += kTile6MC) {
            const int64_t mb_rows = std::min(kTile6MC, m_full - mb);
            PackGemmAPanelDupAvx2(lhs_data, args.lhs_m_stride, kb, mb, mb_rows,
                                  kc_len, buf_a);
            for (int64_t nb = 0; nb < n_full; nb += kTile6NC) {
                const int64_t nb_len = std::min(kTile6NC, n_full - nb);
                for (int64_t sub = 0; sub < nb_len; sub += kBlockedNR) {
                    const auto* const b_source =
                            rhs_data +
                            static_cast<size_t>((nb + sub) * args.rhs_n_stride + kb) *
                                    sizeof(float);
                    PackGemmBPanelAvx2<F32GemmWeightOps, false>(
                            b_source, rhs_row_bytes, kc_len,
                            buf_b + sub * kTile6KC);
                }

                for (int64_t row = 0; row < mb_rows; row += kTile6MR) {
                    for (int64_t sub = 0; sub < nb_len; sub += kBlockedNR) {
                        AccumulateF32Tile6x16DupAvx2(
                                buf_a + row * kc_len * 8,
                                buf_b + sub * kTile6KC, kc_len,
                                args.output_m_stride,
                                output_data + (mb + row) * args.output_m_stride + nb + sub,
                                kc_len * 8);
                    }
                }
            }
        }
    }
}

template<int Rows>
void RunNContiguousTileAvx2(const float* lhs, const float* rhs, float* output,
                            const GemmArgs& args, int64_t depth, bool first_panel) noexcept {
    __m256 acc[Rows][2]{};
    if (!first_panel) {
        for (int row = 0; row < Rows; ++row) {
            const float* dst = output + row * args.output_m_stride;
            acc[row][0] = _mm256_loadu_ps(dst);
            acc[row][1] = _mm256_loadu_ps(dst + 8);
        }
    }

    for (int64_t k = 0; k < depth; ++k) {
        const float* weights = rhs + k * args.rhs_k_stride;
        const __m256 low = _mm256_loadu_ps(weights);
        const __m256 high = _mm256_loadu_ps(weights + 8);
        for (int row = 0; row < Rows; ++row) {
            const __m256 input = _mm256_broadcast_ss(lhs + row * args.lhs_m_stride + k);
            acc[row][0] = _mm256_fmadd_ps(input, low, acc[row][0]);
            acc[row][1] = _mm256_fmadd_ps(input, high, acc[row][1]);
        }
    }

    for (int row = 0; row < Rows; ++row) {
        float* dst = output + row * args.output_m_stride;
        _mm256_storeu_ps(dst, acc[row][0]);
        _mm256_storeu_ps(dst + 8, acc[row][1]);
    }
}

Status RunRhsNContiguousAvx2(const GemmArgs& args) noexcept {
    const auto* lhs = reinterpret_cast<const float*>(args.lhs);
    const auto* rhs = reinterpret_cast<const float*>(args.rhs);
    auto* output = reinterpret_cast<float*>(args.output);
    const int64_t n_full = args.n - args.n % kBlockedNR;
    // Limit the number of strided weight rows touched before advancing N.
    for (int64_t kb = 0; kb < args.k;) {
        constexpr int64_t kKPanel = 128;
        const int64_t depth = std::min(kKPanel, args.k - kb);
        for (int64_t col = 0; col < n_full; col += kBlockedNR) {
            const float* weights = rhs + kb * args.rhs_k_stride + col;
            int64_t row = 0;
            while (row + 4 <= args.m) {
                RunNContiguousTileAvx2<4>(lhs + row * args.lhs_m_stride + kb,
                                          weights,
                                          output + row * args.output_m_stride + col,
                                          args, depth, kb == 0);
                row += 4;
            }

            if (row + 2 <= args.m) {
                RunNContiguousTileAvx2<2>(lhs + row * args.lhs_m_stride + kb,
                                          weights,
                                          output + row * args.output_m_stride + col,
                                          args, depth, kb == 0);
                row += 2;
            }

            if (row < args.m) {
                RunNContiguousTileAvx2<1>(lhs + row * args.lhs_m_stride + kb,
                                          weights,
                                          output + row * args.output_m_stride + col,
                                          args, depth, kb == 0);
            }
        }
        kb += depth;
    }

    if (n_full == args.n) {
        return Status::Ok();
    }

    GemmArgs tail = args;
    tail.n -= n_full;
    tail.rhs = reinterpret_cast<const std::byte*>(rhs + n_full);
    tail.output = reinterpret_cast<std::byte*>(output + n_full);
    return RunGemmA32W32Scalar(tail);
}

// Shared blocked-branch dispatch for the 6x16 candidates: zero the aligned
// M x N core, run the blocked driver, then cover the N-tail and M-tail with
// the row-pair and single-row cores (overwrite semantics).
Status RunBlockedDispatchTile6(const GemmArgs& args, int64_t m_full, int64_t n_full,
                               void (*blocked)(const GemmArgs&, int64_t, int64_t)) noexcept {
    auto* const output_data = reinterpret_cast<float*>(args.output);

    for (int64_t row = 0; row < m_full; ++row) {
        float* out_row = output_data + row * args.output_m_stride;
        for (int64_t col = 0; col < n_full; ++col) {
            out_row[col] = 0.0f;
        }
    }

    blocked(args, m_full, n_full);
    if (n_full < args.n) {
        RunKContiguousRowPairsAvx2(args, 0, m_full, n_full, args.n - n_full);
    }

    if (m_full < args.m) {
        RunKContiguousRowPairsAvx2(args, m_full, args.m, 0, args.n);
    }
    return Status::Ok();
}

Status RunRhsKContiguousAvx2(const GemmArgs& args) noexcept {
    constexpr int64_t kMaxColumnBlockedRows = 9;
    constexpr int64_t kMinColumnBlockedK = 128;
    if (args.m > kAvx2RowsPerPair && args.m <= kMaxColumnBlockedRows &&
        args.k >= kMinColumnBlockedK && args.n > kBlockedNR) {
        // Share each group of 16 weight rows across all row pairs and the odd
        // row before advancing N. Short K does too little work per group to
        // amortize its extra calls.
        int64_t col = 0;
        while (col < args.n) {
            const int64_t delta = std::min(kBlockedNR, args.n - col);
            RunKContiguousRowPairsAvx2(args, 0, args.m, col, delta);
            col += delta;
        }
        return Status::Ok();
    }

    if (args.m <= 8 || args.n < kBlockedNR) {
        RunKContiguousRowPairsAvx2(args, 0, args.m, 0, args.n);
        return Status::Ok();
    }

    // On wide projections, reuse the packed weights for an odd row to avoid
    // rereading the full RHS through the single-row dot-product driver.
    const bool block_odd_row = args.k >= 2048 && args.n >= 1024;
    const int64_t m_full = block_odd_row ? args.m : args.m - args.m % 2;
    const int64_t n_full = args.n - args.n % kBlockedNR;
    return RunBlockedDispatchTile6(args, m_full, n_full, &RunBlockedGemmTile6Avx2);
}

#endif

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGemmA32W32Avx2Fma(const GemmArgs& args) noexcept {
    if (args.m <= 0 || args.n <= 0 || args.k <= 0 || args.lhs_k_stride != 1 ||
        args.output_n_stride != 1) {
        return RunGemmA32W32Scalar(args);
    }

    if (args.rhs_k_stride == 1) {
        return RunRhsKContiguousAvx2(args);
    }

    if (args.rhs_n_stride == 1 && CanUseF32NContiguousAvx2(args.m, args.k, args.n)) {
        return RunRhsNContiguousAvx2(args);
    }
    return RunGemmA32W32Scalar(args);
}

Status RunGemmA32W32Avx2FmaTile6(const GemmArgs& args) noexcept {
    if (args.m >= 9 && args.k > 0 && args.n > 0 &&
        args.lhs_k_stride == 1 && args.rhs_k_stride == 1 &&
        args.output_n_stride == 1) {
        // Even row count so the 6/4/2-row tile decomposition covers every
        // blocked row; at most one odd row remains for the single-row core.
        const int64_t m_full = args.m - args.m % 2;
        const int64_t n_full = args.n - args.n % kBlockedNR;
        return RunBlockedDispatchTile6(args, m_full, n_full, &RunBlockedGemmTile6Avx2);
    }
    return RunGemmA32W32Scalar(args);
}

Status RunGemmA32W32Avx2FmaTile6DupA(const GemmArgs& args) noexcept {
    if (args.m >= 9 && args.k > 0 && args.n > 0 &&
        args.lhs_k_stride == 1 && args.rhs_k_stride == 1 &&
        args.output_n_stride == 1) {
        const int64_t m_full = args.m - args.m % kTile6MR;
        const int64_t n_full = args.n - args.n % kBlockedNR;
        return RunBlockedDispatchTile6(args, m_full, n_full,
                                       &RunBlockedGemmTile6DupAAvx2);
    }
    return RunGemmA32W32Scalar(args);
}
#endif

} // namespace aethermind::cpu::detail
