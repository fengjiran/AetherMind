/// @file gemm_a32_w32_packed_b_avx2.cpp
/// @brief F32 GEMM consuming the exact KC=512, NR=16 packed-B artifact.
///
/// Checked entries validate the borrowed artifact and select a driver.
/// The AVX2 drivers consume validated slices, including blocked edge regions.

#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/base/macros.h"
#include "gemm_a32_w32_packed_b_internal.h"
#include "gemm_a32_w32_policy.h"
#include "gemm_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include "gemm_avx2_internal.h"
#endif

namespace aethermind::cpu::detail {
namespace {

// ---- Blocking constants ---------------------------------------------------

constexpr int64_t kPackedBMB = 48;
constexpr int64_t kPackedBNR = cpu::kCpuBPanelF32Kc512Nr16NR;
constexpr int64_t kPackedBKC = cpu::kCpuBPanelF32Kc512Nr16KC;

#if defined(__AVX2__) && defined(__FMA__)
static_assert(kPackedBMB % 2 == 0);
static_assert(kPackedBKC == kGemmAvx2KC);
static_assert(kPackedBNR == kGemmAvx2NR);

// ---- L2 A-panel packing ---------------------------------------------------

// Copies `mb_rows` A rows of the current KC panel into a [mb_rows][KC] buffer.
// Only [0, kc_len) K columns are written; the caller keeps buf_a aligned.
void PackF32APanelKc512(const float* lhs,
                        int64_t lhs_m_stride,
                        int64_t kb,
                        int64_t mb_base,
                        int64_t mb_rows,
                        int64_t kc_len,
                        float* buf_a) noexcept {
    AM_DCHECK(kc_len <= kPackedBKC);
    for (int64_t row = 0; row < mb_rows; ++row) {
        const float* const source = lhs + (mb_base + row) * lhs_m_stride + kb;
        float* const destination = buf_a + row * kPackedBKC;
        for (int64_t kk = 0; kk < kc_len; ++kk) {
            destination[kk] = source[kk];
        }
    }
}

// ---- L3 drivers -----------------------------------------------------------

// Scans a validated slice with K > 0 and unit output N stride. Scalar edges
// accumulate in double, matching the packed reference semantics.
void RunPackedBScanValidated(const PackedGemmF32Args& packed) noexcept {
    const GemmArgs& args = packed.gemm;
    const float* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    float* const output_data = reinterpret_cast<float*>(args.output);
    const float* const packed_b_data = reinterpret_cast<const float*>(packed.packed_b);

    for (int64_t row = 0; row < args.m; ++row) {
        const float* const lhs_row = lhs_data + row * args.lhs_m_stride;
        float* const output_row = output_data + row * args.output_m_stride;
        int64_t col = 0;
        for (; col < args.n &&
               (packed.weight_n_offset + col) % kPackedBNR != 0;
             ++col) {
            double sum = 0.0;
            const int64_t weight_n = packed.weight_n_offset + col;
            for (int64_t k = 0; k < args.k; ++k) {
                sum += static_cast<double>(lhs_row[k * args.lhs_k_stride]) *
                       static_cast<double>(packed_b_data[cpu::CpuBPanelF32Kc512Nr16ElementIndex(
                               weight_n, k, packed.n_blocks)]);
            }
            output_row[col] = static_cast<float>(sum);
        }

        for (; col + kPackedBNR <= args.n; col += kPackedBNR) {
            const int64_t weight_n = packed.weight_n_offset + col;
            const size_t n_block = static_cast<size_t>(weight_n / kPackedBNR);
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int64_t kb = 0;
            size_t k_panel = 0;
            while (kb < args.k) {
                const int64_t kc_len = std::min(kPackedBKC, args.k - kb);
                const size_t panel_index =
                        (k_panel * static_cast<size_t>(packed.n_blocks) + n_block) *
                        static_cast<size_t>(kPackedBKC * kPackedBNR);
                const float* const b_panel = packed_b_data + panel_index;
                for (int64_t kk = 0; kk < kc_len; ++kk) {
                    const __m256 x = _mm256_broadcast_ss(
                            lhs_row + (kb + kk) * args.lhs_k_stride);
                    const float* const b_row = b_panel + kk * kPackedBNR;
                    acc0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b_row), acc0);
                    acc1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b_row + 8), acc1);
                }
                kb += kc_len;
                ++k_panel;
            }
            _mm256_storeu_ps(output_row + col, acc0);
            _mm256_storeu_ps(output_row + col + 8, acc1);
        }

        for (; col < args.n; ++col) {
            double sum = 0.0;
            const int64_t weight_n = packed.weight_n_offset + col;
            for (int64_t k = 0; k < args.k; ++k) {
                sum += static_cast<double>(lhs_row[k * args.lhs_k_stride]) *
                       static_cast<double>(packed_b_data[cpu::CpuBPanelF32Kc512Nr16ElementIndex(
                               weight_n, k, packed.n_blocks)]);
            }
            output_row[col] = static_cast<float>(sum);
        }
    }
}

// Executes a validated slice selected for blocked GEMM. The first K panel
// overwrites the core rectangle; later panels accumulate into it. The two edge
// slices inherit the artifact and stay within the validated output bounds.
void RunPackedBBlockedValidated(const PackedGemmF32Args& packed) noexcept {
    const GemmArgs& args = packed.gemm;
    // Even row count so the 6/4/2-row tile decomposition covers every blocked
    // row; at most one odd row remains for the scan M-tail.
    const int64_t m_full = args.m - args.m % 2;
    const int64_t n_full = args.n - args.n % kPackedBNR;
    const float* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    float* const output_data = reinterpret_cast<float*>(args.output);
    const float* const packed_b_data = reinterpret_cast<const float*>(packed.packed_b);

    alignas(32) float buf_a[static_cast<size_t>(kPackedBMB * kPackedBKC)];
    int64_t kb = 0;
    size_t k_panel = 0;
    while (kb < args.k) {
        const int64_t kc_len = std::min(kPackedBKC, args.k - kb);
        const bool accumulate = kb != 0;
        for (int64_t mb = 0; mb < m_full; mb += kPackedBMB) {
            const int64_t mb_rows = std::min(kPackedBMB, m_full - mb);
            PackF32APanelKc512(lhs_data, args.lhs_m_stride, kb, mb, mb_rows,
                               kc_len, buf_a);
            for (int64_t nb = 0; nb < n_full; nb += kPackedBNR) {
                const int64_t weight_n = packed.weight_n_offset + nb;
                const size_t n_block = static_cast<size_t>(weight_n / kPackedBNR);
                const size_t panel_index =
                        (k_panel * static_cast<size_t>(packed.n_blocks) + n_block) *
                        static_cast<size_t>(kPackedBKC * kPackedBNR);
                const float* const b_panel = packed_b_data + panel_index;
                for (int64_t row = 0; row < mb_rows;) {
                    const int64_t remaining = mb_rows - row;
                    AM_DCHECK(remaining % 2 == 0);
                    const float* const a_row = buf_a + row * kPackedBKC;
                    float* const out_row = output_data + (mb + row) * args.output_m_stride + nb;
                    if (remaining >= kGemmAvx2MR6) {
                        AccumulateF32Tile6x16Avx2(a_row, b_panel, kc_len,
                                                  args.output_m_stride, out_row, kPackedBKC,
                                                  accumulate);
                        row += kGemmAvx2MR6;
                    } else if (remaining >= 4) {
                        AccumulateF32Tile4x16Avx2(a_row, b_panel, kc_len,
                                                  args.output_m_stride, out_row, kPackedBKC,
                                                  accumulate);
                        row += 4;
                    } else {
                        AccumulateF32Tile2x16Avx2(a_row, b_panel, kc_len,
                                                  args.output_m_stride, out_row, kPackedBKC,
                                                  accumulate);
                        row += 2;
                    }
                }
            }
        }
        kb += kc_len;
        ++k_panel;
    }

    if (n_full < args.n) {
        PackedGemmF32Args right_edge = packed;
        right_edge.gemm.m = m_full;
        right_edge.gemm.n = args.n - n_full;
        right_edge.gemm.output = reinterpret_cast<std::byte*>(output_data + n_full);
        right_edge.weight_n_offset += n_full;
        RunPackedBScanValidated(right_edge);
    }
    if (m_full < args.m) {
        PackedGemmF32Args bottom_edge = packed;
        bottom_edge.gemm.lhs = reinterpret_cast<const std::byte*>(lhs_data + m_full * args.lhs_m_stride);
        bottom_edge.gemm.output = reinterpret_cast<std::byte*>(output_data + m_full * args.output_m_stride);
        bottom_edge.gemm.m = args.m - m_full;
        RunPackedBScanValidated(bottom_edge);
    }
}

#endif

// ---- L4 entry -------------------------------------------------------------

Status RunPackedBChecked(const PackedGemmF32Args& packed,
                         [[maybe_unused]] bool allow_blocked) noexcept {
    AM_RETURN_IF_ERROR(ValidatePackedBArgs(packed));
    const GemmArgs& args = packed.gemm;
    if (args.m == 0 || args.n == 0) {
        return Status::Ok();
    }
    if (args.output == nullptr) {
        return Status::InvalidArgument("Packed GEMM scan requires output data");
    }
    if (args.k > 0 && args.lhs == nullptr) {
        return Status::InvalidArgument("Packed GEMM scan requires input data");
    }

#if defined(__AVX2__) && defined(__FMA__)
    const PackedBStrategy strategy = PlanPackedB(packed);
    if (allow_blocked && strategy == PackedBStrategy::kBlocked) {
        RunPackedBBlockedValidated(packed);
        return Status::Ok();
    }
    if (strategy != PackedBStrategy::kReference) {
        RunPackedBScanValidated(packed);
        return Status::Ok();
    }
#endif
    RunPackedBReferenceValidated(packed);
    return Status::Ok();
}

} // namespace

Status RunGemmA32W32PackedBScan(const PackedGemmF32Args& packed) noexcept {
    return RunPackedBChecked(packed, false);
}

Status RunGemmA32W32PackedBBlocked(const PackedGemmF32Args& packed) noexcept {
    return RunPackedBChecked(packed, true);
}

Status RunGemmA32W32PackedB(const PackedGemmF32Args& packed) noexcept {
    return RunPackedBChecked(packed, true);
}

} // namespace aethermind::cpu::detail
