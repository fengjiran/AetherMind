#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A32_W16_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A32_W16_AVX2_INTERNAL_H

/// @file gemm_a32_w16_avx2_internal.h
/// @brief AVX2 F32-activation x 16-bit-weight GEMM kernels for the A32W16 family.
///
/// Only translation units compiled with AVX2 and FMA enabled (`-mavx2 -mfma`)
/// may include this header: it pulls in intrinsics-based inline definitions
/// from the shared AVX2 headers, and the F16 weight specialization additionally
/// requires F16C.

#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_avx2_internal.h"
#include "gemm_internal.h"
#include "gemm_w16_avx2_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace aethermind::cpu::detail {

template<typename WeightOps, int Rows, int Cols>
bool TryGemmW16DotTile(const GemmArgs& args, int64_t row, int64_t col,
                       float input_magnitude) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    auto* output = reinterpret_cast<float*>(args.output);
    __m256 acc[Rows][Cols];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            acc[r][c] = _mm256_setzero_ps();
        }
    }

    // A single output row leaves registers for independent magnitude chains.
    // The 2-row tile uses its remaining registers for the extra FMA sums.
    constexpr int kMagnitudeChains = Rows == 1 ? Cols : 1;
    __m256 weight_magnitude[kMagnitudeChains]{};
    const __m256 sign = _mm256_set1_ps(-0.0f);
    int64_t k = 0;
    for (; k + 8 <= args.k; k += 8) {
        __m256 inputs[Rows];
        for (int r = 0; r < Rows; ++r) {
            inputs[r] = _mm256_loadu_ps(input + (row + r) * args.lhs_m_stride + k);
        }

        for (int c = 0; c < Cols; ++c) {
            const __m256 weights = WeightOps::LoadEight(
                    args.rhs + static_cast<size_t>((col + c) * args.rhs_n_stride + k) * sizeof(uint16_t));
            __m256& magnitude = weight_magnitude[c % kMagnitudeChains];
            magnitude = _mm256_max_ps(magnitude, _mm256_andnot_ps(sign, weights));
            for (int r = 0; r < Rows; ++r) {
                acc[r][c] = _mm256_fmadd_ps(inputs[r], weights, acc[r][c]);
            }
        }
    }

    __m256 reduced_magnitude = weight_magnitude[0];
    for (int c = 1; c < kMagnitudeChains; ++c) {
        reduced_magnitude = _mm256_max_ps(reduced_magnitude, weight_magnitude[c]);
    }

    alignas(32) float magnitudes[8];
    _mm256_store_ps(magnitudes, reduced_magnitude);
    float max_weight = 0.0f;
    for (const float value: magnitudes) {
        max_weight = std::max(max_weight, value);
    }

    float results[Rows][Cols];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            results[r][c] = HorizontalSumAvx2(acc[r][c]);
        }
    }

    for (; k < args.k; ++k) {
        for (int c = 0; c < Cols; ++c) {
            const float weight = WeightOps::LoadOne(
                    args.rhs + static_cast<size_t>((col + c) * args.rhs_n_stride + k) *
                                       sizeof(uint16_t));
            max_weight = std::max(max_weight, std::fabs(weight));
            for (int r = 0; r < Rows; ++r) {
                results[r][c] += input[(row + r) * args.lhs_m_stride + k] * weight;
            }
        }
    }

    if (!GemmReductionMagnitudeIsSafe(input_magnitude, max_weight, args.k)) {
        return false;
    }

    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            if (!std::isfinite(results[r][c])) {
                return false;
            }
            output[(row + r) * args.output_m_stride + col + c] = results[r][c];
        }
    }
    return true;
}

template<typename WeightOps, int Rows>
bool TryGemmW16Rows(const GemmArgs& args, int64_t row) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    float max_input = 0.0f;
    for (int r = 0; r < Rows; ++r) {
        max_input = std::max(
                max_input,
                GemmInputMagnitude(input + (row + r) * args.lhs_m_stride, args.k));
    }

    int64_t col = 0;
    for (; col + 4 <= args.n; col += 4) {
        if (!TryGemmW16DotTile<WeightOps, Rows, 4>(args, row, col, max_input)) {
            return false;
        }
    }

    for (; col < args.n; ++col) {
        if (!TryGemmW16DotTile<WeightOps, Rows, 1>(args, row, col, max_input)) {
            return false;
        }
    }
    return true;
}

template<typename WeightOps>
bool TryGemmW16Direct(const GemmArgs& args) noexcept {
    int64_t row = 0;
    for (; row + 2 <= args.m; row += 2) {
        if (!TryGemmW16Rows<WeightOps, 2>(args, row)) {
            return false;
        }
    }
    return row == args.m || TryGemmW16Rows<WeightOps, 1>(args, row);
}

template<typename WeightOps, bool PackA>
bool TryGemmW16FullTiles(const GemmArgs& args, int64_t m_full, int64_t n_full) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    auto* output = reinterpret_cast<float*>(args.output);

    // Reuse each B panel across a larger M block. Cap the logical direct A
    // footprint at 512 KiB per K panel to limit repeated reads and page walks;
    // 252 is the largest multiple of the 6-row tile below that cap.
    constexpr int64_t kDirectRows = 252;
    static_assert(kDirectRows % kGemmAvx2MR6 == 0);
    const int64_t row_block = PackA ? kGemmAvx2MB : kDirectRows;
    const int64_t a_stride = PackA ? kGemmAvx2KC : args.lhs_m_stride;
    alignas(32) float panel_a[PackA ? kGemmAvx2MB * kGemmAvx2KC : 1];
    alignas(32) float panel_b[kGemmAvx2KC * kGemmAvx2NR];
    const size_t weight_row_bytes = static_cast<size_t>(args.rhs_n_stride) * sizeof(uint16_t);
    for (int64_t kb = 0; kb < args.k; kb += kGemmAvx2KC) {
        const int64_t kc = std::min(kGemmAvx2KC, args.k - kb);
        for (int64_t mb = 0; mb < m_full; mb += row_block) {
            // Both row_block candidates are even and m_full is rounded to an
            // even count, so every block decomposes into whole 6/4/2-row tiles.
            const int64_t rows = std::min(row_block, m_full - mb);
            const float* src_a = input + mb * args.lhs_m_stride + kb;
            float max_a = 0.0f;
            for (int64_t row = 0; row < rows; ++row) {
                const float* src = src_a + row * args.lhs_m_stride;
                if constexpr (PackA) {
                    std::copy_n(src, kc, panel_a + row * kGemmAvx2KC);
                }
                max_a = std::max(max_a, GemmInputMagnitude(src, kc));
            }

            const float* a = PackA ? panel_a : src_a;
            for (int64_t nb = 0; nb < n_full; nb += kGemmAvx2NR) {
                const auto* src = args.rhs +
                                  static_cast<size_t>(nb * args.rhs_n_stride + kb) * sizeof(uint16_t);
                if (const float max_b = PackGemmW16BPanelAvx2<WeightOps>(
                            src, weight_row_bytes, kc, panel_b);
                    !GemmReductionMagnitudeIsSafe(max_a, max_b, args.k)) {
                    return false;
                }

                for (int64_t row = 0; row < rows;) {
                    const int64_t remaining = rows - row;
                    const float* a_row = a + row * a_stride;
                    float* const out_row = output + (mb + row) * args.output_m_stride + nb;
                    if (remaining >= kGemmAvx2MR6) {
                        AccumulateF32Tile6x16Avx2(a_row, panel_b, kc, args.output_m_stride,
                                                  out_row, a_stride);
                        row += kGemmAvx2MR6;
                    } else if (remaining >= 4) {
                        AccumulateF32Tile4x16UnrolledAvx2(a_row, panel_b, kc, args.output_m_stride,
                                                          out_row, a_stride);
                        row += 4;
                    } else {
                        AccumulateF32Tile2x16UnrolledAvx2(a_row, panel_b, kc, args.output_m_stride,
                                                          out_row, a_stride);
                        row += 2;
                    }
                }
            }
        }
    }
    return true;
}

template<typename WeightOps>
bool TryGemmW16Blocked(const GemmArgs& args) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    auto* output = reinterpret_cast<float*>(args.output);
    // Even row count so the 6/4/2-row tile decomposition covers every blocked
    // row; at most one odd row remains for the direct M-tail path.
    const int64_t m_full = args.m - args.m % 2;
    const int64_t n_full = args.n - args.n % kGemmAvx2NR;
    if (n_full == 0) {
        return TryGemmW16Direct<WeightOps>(args);
    }

    for (int64_t row = 0; row < m_full; ++row) {
        std::fill_n(output + row * args.output_m_stride, n_full, 0.0f);
    }

    // A short K, such as the narrow-K LM head in the model fixture, has little
    // B-packing cost and benefits from the smaller C working set of packed A.
    const bool reuse_b = m_full > kGemmAvx2MB && args.k >= kGemmAvx2KC;
    const bool computed = reuse_b
                                  ? TryGemmW16FullTiles<WeightOps, false>(args, m_full, n_full)
                                  : TryGemmW16FullTiles<WeightOps, true>(args, m_full, n_full);
    if (!computed) {
        return false;
    }

    for (int64_t row = 0; row < m_full; ++row) {
        for (int64_t col = 0; col < n_full; ++col) {
            if (!std::isfinite(output[row * args.output_m_stride + col])) {
                return false;
            }
        }
    }

    if (n_full < args.n) {
        GemmArgs edge = args;
        edge.m = m_full;
        edge.n -= n_full;
        edge.rhs += static_cast<size_t>(n_full * args.rhs_n_stride) * sizeof(uint16_t);
        edge.output = reinterpret_cast<std::byte*>(output + n_full);
        if (!TryGemmW16Direct<WeightOps>(edge)) {
            return false;
        }
    }

    if (m_full < args.m) {
        GemmArgs edge = args;
        edge.m -= m_full;
        edge.lhs = reinterpret_cast<const std::byte*>(input + m_full * args.lhs_m_stride);
        edge.output = reinterpret_cast<std::byte*>(output + m_full * args.output_m_stride);
        if (!TryGemmW16Direct<WeightOps>(edge)) {
            return false;
        }
    }
    return true;
}

template<typename WeightOps>
bool TryGemmA32W16Avx2Fma(const GemmArgs& args) noexcept {
    if (args.m == 0 || args.n == 0 || args.k == 0 ||
        args.lhs_k_stride != 1 || args.rhs_k_stride != 1 || args.output_n_stride != 1) {
        return false;
    }
    return args.m <= 8 ? TryGemmW16Direct<WeightOps>(args) : TryGemmW16Blocked<WeightOps>(args);
}

} // namespace aethermind::cpu::detail
#endif
