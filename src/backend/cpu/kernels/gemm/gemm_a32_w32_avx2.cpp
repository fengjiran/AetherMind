/// @file gemm_a32_w32_avx2.cpp
/// @brief F32 (A32W32) AVX2 GEMM entry, blocking drivers and tile primitives.
///
/// Layer map (dependencies point downward only):
///   L4 entry/orchestration: RunGemmA32W32Avx2Fma; owns zero-fill, tails
///      and the only scalar fallback.
///   L3 drivers: panel/rectangle loops over pre-described extents; never call
///      scalar or reference fallbacks.
///   L2 primitives: single-tile intrinsics; no Status, allocation or fallback.
///   L1 policy: F32Avx2Plan / PlanF32Avx2 in gemm_a32_w32_policy.h.
///   L0 contract: GemmArgs in gemm_internal.h.
/// The DupA measurement candidate lives in
/// tests/benchmark/cpu_kernels/gemm_avx2_candidates.cpp.

#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "aethermind/base/macros.h"
#include "backend/cpu/kernels/dot_product/dot_product_internal.h"
#include "gemm_a32_w32_policy.h"
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
constexpr int64_t kBlockedNR = kGemmAvx2NR;
constexpr int64_t kBlockedKC = kGemmAvx2KC;
constexpr int64_t kTile6MR = kGemmAvx2MR6;

static_assert(kF32Avx2NR == kGemmAvx2NR);

// NOLINTBEGIN(portability-simd-intrinsics)

// ---- L2 primitives --------------------------------------------------------
// Single-tile intrinsics; callers pass validated typed pointers and strides.

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

// Two rows share each of four weight vectors across eight independent sums.
void OverwriteF32RowPairAvx2(const float* lhs0, const float* lhs1, const float* rhs,
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

// Rows x 16 N-contiguous tile over one K panel. The first panel overwrites the
// output; later panels accumulate into it.
template<int Rows>
void AccumulateF32NContiguousTileAvx2(const float* lhs, const float* rhs, float* output,
                                      int64_t lhs_m_stride, int64_t rhs_k_stride,
                                      int64_t output_m_stride, int64_t depth,
                                      bool first_panel) noexcept {
    __m256 acc[Rows][2]{};
    if (!first_panel) {
        for (int row = 0; row < Rows; ++row) {
            const float* dst = output + row * output_m_stride;
            acc[row][0] = _mm256_loadu_ps(dst);
            acc[row][1] = _mm256_loadu_ps(dst + 8);
        }
    }

    for (int64_t k = 0; k < depth; ++k) {
        const float* weights = rhs + k * rhs_k_stride;
        const __m256 low = _mm256_loadu_ps(weights);
        const __m256 high = _mm256_loadu_ps(weights + 8);
        for (int row = 0; row < Rows; ++row) {
            const __m256 input = _mm256_broadcast_ss(lhs + row * lhs_m_stride + k);
            acc[row][0] = _mm256_fmadd_ps(input, low, acc[row][0]);
            acc[row][1] = _mm256_fmadd_ps(input, high, acc[row][1]);
        }
    }

    for (int row = 0; row < Rows; ++row) {
        float* dst = output + row * output_m_stride;
        _mm256_storeu_ps(dst, acc[row][0]);
        _mm256_storeu_ps(dst + 8, acc[row][1]);
    }
}

// ---- L3 drivers -----------------------------------------------------------
// Blocking over K/N/M extents. No scalar/reference fallback and no capability
// checks; the entry or an L4 helper owns both.

// The production 6x16 micro-kernel keeps twelve independent accumulator
// chains. The 6/4/2-row decomposition keeps complete row pairs on the same B
// panel. Candidates can add pack variants without changing this driver.
void RunBlockedTile6DriverAvx2(const GemmArgs& args, int64_t m_full, int64_t n_full) noexcept {
    AM_DCHECK(m_full >= 0 && m_full <= args.m);
    AM_DCHECK(n_full >= 0 && n_full <= args.n && n_full % kBlockedNR == 0);
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const std::byte*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);
    const size_t rhs_row_bytes = static_cast<size_t>(args.rhs_n_stride) * sizeof(float);
    const int64_t paired_rows = m_full - m_full % 2;

    alignas(32) float buf_b[static_cast<size_t>(kBlockedKC * kBlockedNR)];

    for (int64_t kb = 0; kb < args.k; kb += kBlockedKC) {
        const int64_t kc_len = std::min(kBlockedKC, args.k - kb);
        for (int64_t nb = 0; nb < n_full; nb += kBlockedNR) {
            const auto* const b_src =
                    rhs_data +
                    static_cast<size_t>(nb * args.rhs_n_stride + kb) * sizeof(float);
            PackGemmBPanelAvx2<F32GemmWeightOps, false>(b_src, rhs_row_bytes, kc_len, buf_b);
            for (int64_t row = 0; row < paired_rows;) {
                if (const int64_t remaining = paired_rows - row; remaining >= kTile6MR) {
                    AccumulateDirectF32Tile6x16Avx2(
                            lhs_data + row * args.lhs_m_stride + kb,
                            buf_b, kc_len, args.output_m_stride,
                            output_data + row * args.output_m_stride + nb,
                            args.lhs_m_stride);
                    row += kTile6MR;
                } else if (remaining >= 4) {
                    AccumulateF32Tile4x16Avx2(
                            lhs_data + row * args.lhs_m_stride + kb,
                            buf_b, kc_len, args.output_m_stride,
                            output_data + row * args.output_m_stride + nb,
                            args.lhs_m_stride);
                    row += 4;
                } else {
                    AccumulateF32Tile2x16Avx2(
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
                        [=](int, int64_t k) {
                            return _mm256_broadcast_ss(lhs_row + k);
                        });
            }
        }
    }
}

// Overwrites a nonempty rectangle directly from K-contiguous weight rows.
void RunRowPairRectDriverAvx2(const GemmArgs& args, int64_t row_begin, int64_t row_end,
                              int64_t col_begin, int64_t cols) noexcept {
    AM_DCHECK(row_begin < row_end);
    AM_DCHECK(cols > 0);
    const auto* lhs = reinterpret_cast<const float*>(args.lhs);
    const auto* rhs = reinterpret_cast<const float*>(args.rhs) + col_begin * args.rhs_n_stride;
    auto* output = reinterpret_cast<float*>(args.output) + col_begin;
    int64_t row = row_begin;
    while (row + kAvx2RowsPerPair <= row_end) {
        OverwriteF32RowPairAvx2(lhs + row * args.lhs_m_stride,
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

// Covers the 16-aligned N core only; the caller owns the N tail.
void RunNContiguousDriverAvx2(const GemmArgs& args, int64_t n_full) noexcept {
    AM_DCHECK(n_full >= 0 && n_full <= args.n && n_full % kBlockedNR == 0);
    const auto* lhs = reinterpret_cast<const float*>(args.lhs);
    const auto* rhs = reinterpret_cast<const float*>(args.rhs);
    auto* output = reinterpret_cast<float*>(args.output);
    // Limit the number of strided weight rows touched before advancing N.
    for (int64_t kb = 0; kb < args.k;) {
        constexpr int64_t kKPanel = 128;
        const int64_t depth = std::min(kKPanel, args.k - kb);
        for (int64_t col = 0; col < n_full; col += kBlockedNR) {
            const float* weights = rhs + kb * args.rhs_k_stride + col;
            int64_t row = 0;
            while (row + 4 <= args.m) {
                AccumulateF32NContiguousTileAvx2<4>(
                        lhs + row * args.lhs_m_stride + kb, weights,
                        output + row * args.output_m_stride + col,
                        args.lhs_m_stride, args.rhs_k_stride,
                        args.output_m_stride, depth, kb == 0);
                row += 4;
            }

            if (row + 2 <= args.m) {
                AccumulateF32NContiguousTileAvx2<2>(
                        lhs + row * args.lhs_m_stride + kb, weights,
                        output + row * args.output_m_stride + col,
                        args.lhs_m_stride, args.rhs_k_stride,
                        args.output_m_stride, depth, kb == 0);
                row += 2;
            }

            if (row < args.m) {
                AccumulateF32NContiguousTileAvx2<1>(
                        lhs + row * args.lhs_m_stride + kb, weights,
                        output + row * args.output_m_stride + col,
                        args.lhs_m_stride, args.rhs_k_stride,
                        args.output_m_stride, depth, kb == 0);
            }
        }
        kb += depth;
    }
}

// ---- L4 orchestration -----------------------------------------------------

// Zeroes the aligned core, accumulates it through RunCore, then overwrites the
// N and M tails with the row-pair driver. Keep it out of line to limit
// code size in the dispatcher.
template<void (*RunCore)(const GemmArgs&, int64_t, int64_t) noexcept>
AM_NOINLINE Status RunBlockedWithRowPairTailsAvx2(const GemmArgs& args, int64_t m_full,
                                                  int64_t n_full) noexcept {
    AM_DCHECK(m_full >= 0 && m_full <= args.m);
    AM_DCHECK(n_full >= 0 && n_full <= args.n);
    auto* const output_data = reinterpret_cast<float*>(args.output);

    for (int64_t row = 0; row < m_full; ++row) {
        float* out_row = output_data + row * args.output_m_stride;
        for (int64_t col = 0; col < n_full; ++col) {
            out_row[col] = 0.0f;
        }
    }

    RunCore(args, m_full, n_full);
    if (n_full < args.n) {
        RunRowPairRectDriverAvx2(args, 0, m_full, n_full, args.n - n_full);
    }

    if (m_full < args.m) {
        RunRowPairRectDriverAvx2(args, m_full, args.m, 0, args.n);
    }
    return Status::Ok();
}

Status RunKContiguousPlanAvx2(const GemmArgs& args, const F32Avx2Plan& plan) noexcept {
    if (plan.strategy == F32Avx2Strategy::kRowPairs) {
        RunRowPairRectDriverAvx2(args, 0, args.m, 0, args.n);
        return Status::Ok();
    }

    if (plan.strategy == F32Avx2Strategy::kColumnBlockedRowPairs) {
        int64_t col = 0;
        while (col < args.n) {
            const int64_t cols = std::min(kBlockedNR, args.n - col);
            RunRowPairRectDriverAvx2(args, 0, args.m, col, cols);
            col += cols;
        }
        return Status::Ok();
    }

    AM_DCHECK(plan.strategy == F32Avx2Strategy::kBlockedTile6);
    return RunBlockedWithRowPairTailsAvx2<RunBlockedTile6DriverAvx2>(
            args, plan.m_full, plan.n_full);
}

Status RunNContiguousPlanAvx2(const GemmArgs& args, int64_t n_full) noexcept {
    RunNContiguousDriverAvx2(args, n_full);
    if (n_full == args.n) {
        return Status::Ok();
    }

    GemmArgs tail = args;
    tail.n -= n_full;
    tail.rhs = reinterpret_cast<const std::byte*>(
            reinterpret_cast<const float*>(args.rhs) + n_full);
    tail.output = reinterpret_cast<std::byte*>(
            reinterpret_cast<float*>(args.output) + n_full);
    return RunGemmA32W32Scalar(tail);
}

// NOLINTEND(portability-simd-intrinsics)
#endif

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGemmA32W32Avx2Fma(const GemmArgs& args) noexcept {
    const F32Avx2Plan plan = PlanF32Avx2(args);
    if (plan.layout == F32Avx2Layout::kRhsKContiguous) {
        return RunKContiguousPlanAvx2(args, plan);
    }

    if (plan.layout == F32Avx2Layout::kRhsNContiguous) {
        return RunNContiguousPlanAvx2(args, plan.n_full);
    }
    return RunGemmA32W32Scalar(args);
}
#endif

} // namespace aethermind::cpu::detail
