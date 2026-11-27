/// @file gemm_avx2_candidates.cpp
/// @brief Test/benchmark-only A32W32 AVX2 measurement candidates.
///
/// Not part of AetherMind: this translation unit is compiled only into the
/// unit-test and benchmark executables. DupA packs A in broadcast layout (each
/// scalar replicated 8x) so the inner loop runs pure vector loads and FMAs. It
/// uses KC=256 panels so the duplicated A block (48 rows x 256 x 8 floats) and
/// the B block (128 cols x 256 floats) stay L2-resident and amortize both
/// packs. Measured on Alder Lake it loses to the production direct-broadcast
/// kernel, so it stays unregistered.

#include "aethermind/base/macros.h"
#include "backend/cpu/kernels/gemm/gemm_internal.h"
#include "gemm_avx2_candidate_entries.h"

#include <algorithm>
#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include "backend/cpu/kernels/gemm/gemm_avx2_internal.h"
#include <immintrin.h>
#endif

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__)

constexpr int64_t kCandidateNR = kGemmAvx2NR;
constexpr int64_t kCandidateMR6 = kGemmAvx2MR6;
constexpr int64_t kCandidateKC = 256;
constexpr int64_t kCandidateMC = 48;
constexpr int64_t kCandidateNC = 128;

static_assert(kCandidateNR == 16);

// NOLINTBEGIN(portability-simd-intrinsics)

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

void RunBlockedTile6DupADriverAvx2(const GemmArgs& args, int64_t m_full,
                                   int64_t n_full) noexcept {
    const auto* const lhs_data = reinterpret_cast<const float*>(args.lhs);
    const auto* const rhs_data = reinterpret_cast<const std::byte*>(args.rhs);
    auto* const output_data = reinterpret_cast<float*>(args.output);
    const size_t rhs_row_bytes = static_cast<size_t>(args.rhs_n_stride) * sizeof(float);

    // Both panels are reused across the other operand's block loop, which is
    // what lets the duplicated A layout amortize its 8x size amplification.
    alignas(32) float buf_a[static_cast<size_t>(kCandidateMC * kCandidateKC * 8)];
    alignas(32) float buf_b[static_cast<size_t>(kCandidateNC * kCandidateKC)];

    for (int64_t kb = 0; kb < args.k; kb += kCandidateKC) {
        const int64_t kc_len = std::min(kCandidateKC, args.k - kb);
        for (int64_t mb = 0; mb < m_full; mb += kCandidateMC) {
            const int64_t mb_rows = std::min(kCandidateMC, m_full - mb);
            PackGemmAPanelDupAvx2(lhs_data, args.lhs_m_stride, kb, mb, mb_rows,
                                  kc_len, buf_a);
            for (int64_t nb = 0; nb < n_full; nb += kCandidateNC) {
                const int64_t nb_len = std::min(kCandidateNC, n_full - nb);
                for (int64_t sub = 0; sub < nb_len; sub += kCandidateNR) {
                    const auto* const b_source =
                            rhs_data +
                            static_cast<size_t>((nb + sub) * args.rhs_n_stride + kb) *
                                    sizeof(float);
                    PackGemmBPanelAvx2<F32GemmWeightOps, false>(
                            b_source, rhs_row_bytes, kc_len,
                            buf_b + sub * kCandidateKC);
                }

                for (int64_t row = 0; row < mb_rows; row += kCandidateMR6) {
                    for (int64_t sub = 0; sub < nb_len; sub += kCandidateNR) {
                        AccumulateF32Tile6x16DupAvx2(
                                buf_a + row * kc_len * 8,
                                buf_b + sub * kCandidateKC, kc_len,
                                args.output_m_stride,
                                output_data + (mb + row) * args.output_m_stride + nb + sub,
                                kc_len * 8);
                    }
                }
            }
        }
    }
}

// NOLINTEND(portability-simd-intrinsics)
#endif

} // namespace

#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
Status RunGemmA32W32Avx2FmaTile6DupA(const GemmArgs& args) noexcept {
#if !defined(__AVX2__) || !defined(__FMA__)
    return RunGemmA32W32Scalar(args);
#else
    if (args.m < 9 || args.k <= 0 || args.n <= 0 ||
        args.lhs_k_stride != 1 || args.rhs_k_stride != 1 ||
        args.output_n_stride != 1) {
        return RunGemmA32W32Scalar(args);
    }

    const int64_t m_full = args.m - args.m % kCandidateMR6;
    const int64_t n_full = args.n - args.n % kCandidateNR;
    auto* const output = reinterpret_cast<float*>(args.output);
    for (int64_t row = 0; row < m_full; ++row) {
        std::fill_n(output + row * args.output_m_stride, n_full, 0.0F);
    }
    RunBlockedTile6DupADriverAvx2(args, m_full, n_full);

    // Tails reuse the production entry. With rhs_k_stride==1 and these tail
    // extents the plan selects row-pair coverage, matching the former private
    // row-pair tail driver.
    if (n_full < args.n) {
        GemmArgs tail = args;
        tail.m = m_full;
        tail.n = args.n - n_full;
        tail.rhs = reinterpret_cast<const std::byte*>(
                reinterpret_cast<const float*>(args.rhs) + n_full * args.rhs_n_stride);
        tail.output = reinterpret_cast<std::byte*>(output + n_full);
        AM_RETURN_IF_ERROR(RunGemmA32W32Avx2Fma(tail));
    }

    if (m_full < args.m) {
        GemmArgs tail = args;
        tail.m = args.m - m_full;
        tail.lhs = reinterpret_cast<const std::byte*>(
                reinterpret_cast<const float*>(args.lhs) + m_full * args.lhs_m_stride);
        tail.output = reinterpret_cast<std::byte*>(
                output + m_full * args.output_m_stride);
        AM_RETURN_IF_ERROR(RunGemmA32W32Avx2Fma(tail));
    }
    return Status::Ok();
#endif
}
#endif

} // namespace aethermind::cpu::detail