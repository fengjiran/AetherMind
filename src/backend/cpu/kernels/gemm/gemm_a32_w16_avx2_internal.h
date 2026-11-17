#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A32_W16_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_A32_W16_AVX2_INTERNAL_H

#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_avx2_internal.h"
#include "gemm_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace aethermind::cpu::detail {

inline float GemmInputMagnitude(const float* input, int64_t k) noexcept {
    float magnitude = 0.0F;
    for (int64_t i = 0; i < k; ++i) {
        if (!std::isfinite(input[i])) {
            return std::numeric_limits<float>::infinity();
        }
        magnitude = std::max(magnitude, std::fabs(input[i]));
    }
    return magnitude;
}

inline bool GemmReductionMagnitudeIsSafe(float input, float weight, int64_t k) noexcept {
    // Bounding the sum of absolute products also bounds every scalar/SIMD
    // partial sum. Extreme inputs retain the reference's overflow semantics.
    return std::isfinite(input) && std::isfinite(weight) &&
           static_cast<double>(input) * static_cast<double>(weight) * static_cast<double>(k) <=
                   static_cast<double>(std::numeric_limits<float>::max()) * 0.5;
}

template<class WeightOps, int Rows, int Columns>
bool TryGemmW16DotTile(const GemmArgs& args, int64_t row, int64_t col, float input_magnitude) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    auto* output = reinterpret_cast<float*>(args.output);
    __m256 sums[Rows][Columns];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) {
            sums[r][c] = _mm256_setzero_ps();
        }
    }
    __m256 weight_magnitude = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0F);
    int64_t k = 0;
    for (; k + 8 <= args.k; k += 8) {
        __m256 inputs[Rows];
        for (int r = 0; r < Rows; ++r) {
            inputs[r] = _mm256_loadu_ps(input + (row + r) * args.lhs_m_stride + k);
        }
        for (int c = 0; c < Columns; ++c) {
            const __m256 weights = WeightOps::LoadEight(
                    args.rhs + static_cast<size_t>((col + c) * args.rhs_n_stride + k) * sizeof(uint16_t));
            weight_magnitude = _mm256_max_ps(weight_magnitude, _mm256_andnot_ps(sign, weights));
            for (int r = 0; r < Rows; ++r) {
                sums[r][c] = _mm256_fmadd_ps(inputs[r], weights, sums[r][c]);
            }
        }
    }
    alignas(32) float magnitudes[8];
    _mm256_store_ps(magnitudes, weight_magnitude);
    float max_weight = 0.0F;
    for (const float value: magnitudes) {
        max_weight = std::max(max_weight, value);
    }
    float results[Rows][Columns];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Columns; ++c) {
            results[r][c] = HorizontalSumAvx2(sums[r][c]);
        }
    }
    for (; k < args.k; ++k) {
        for (int c = 0; c < Columns; ++c) {
            const float weight = WeightOps::LoadOne(
                    args.rhs + static_cast<size_t>((col + c) * args.rhs_n_stride + k) * sizeof(uint16_t));
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
        for (int c = 0; c < Columns; ++c) {
            if (!std::isfinite(results[r][c])) {
                return false;
            }
            output[(row + r) * args.output_m_stride + col + c] = results[r][c];
        }
    }
    return true;
}

template<class WeightOps, int Rows>
bool TryGemmW16Rows(const GemmArgs& args, int64_t row) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    float max_input = 0.0F;
    for (int r = 0; r < Rows; ++r) {
        max_input = std::max(max_input, GemmInputMagnitude(input + (row + r) * args.lhs_m_stride, args.k));
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

template<class WeightOps>
bool TryGemmW16Direct(const GemmArgs& args) noexcept {
    int64_t row = 0;
    for (; row + 2 <= args.m; row += 2) {
        if (!TryGemmW16Rows<WeightOps, 2>(args, row)) {
            return false;
        }
    }
    return row == args.m || TryGemmW16Rows<WeightOps, 1>(args, row);
}

template<class WeightOps>
bool TryGemmW16Blocked(const GemmArgs& args) noexcept {
    const auto* input = reinterpret_cast<const float*>(args.lhs);
    auto* output = reinterpret_cast<float*>(args.output);
    const int64_t m_full = args.m - args.m % kGemmAvx2MR;
    const int64_t n_full = args.n - args.n % kGemmAvx2NR;
    if (n_full == 0) {
        return TryGemmW16Direct<WeightOps>(args);
    }
    for (int64_t row = 0; row < m_full; ++row) {
        std::fill_n(output + row * args.output_m_stride, n_full, 0.0F);
    }

    // Fixed, bounded per-invocation scratch, shared with the F32 tile shape.
    // Parallel consumers must first budget this as worker scratch.
    alignas(32) float panel_a[kGemmAvx2MB * kGemmAvx2KC];
    alignas(32) float panel_b[kGemmAvx2KC * kGemmAvx2NR];
    for (int64_t kb = 0; kb < args.k; kb += kGemmAvx2KC) {
        const int64_t kc = std::min(kGemmAvx2KC, args.k - kb);
        for (int64_t mb = 0; mb < m_full; mb += kGemmAvx2MB) {
            const int64_t rows = std::min(kGemmAvx2MB, m_full - mb);
            float max_a = 0.0F;
            for (int64_t row = 0; row < rows; ++row) {
                const float* source = input + (mb + row) * args.lhs_m_stride + kb;
                std::copy_n(source, kc, panel_a + row * kGemmAvx2KC);
                max_a = std::max(max_a, GemmInputMagnitude(source, kc));
            }
            for (int64_t nb = 0; nb < n_full; nb += kGemmAvx2NR) {
                __m256 max_b_vector = _mm256_setzero_ps();
                float max_b = 0.0F;
                for (int64_t col = 0; col < kGemmAvx2NR; ++col) {
                    const std::byte* source = args.rhs +
                                              static_cast<size_t>((nb + col) * args.rhs_n_stride + kb) * sizeof(uint16_t);
                    int64_t k = 0;
                    for (; k + 8 <= kc; k += 8) {
                        const __m256 values = WeightOps::LoadEight(source + static_cast<size_t>(k) * sizeof(uint16_t));
                        max_b_vector = _mm256_max_ps(max_b_vector, _mm256_andnot_ps(_mm256_set1_ps(-0.0F), values));
                        alignas(32) float lanes[8];
                        _mm256_store_ps(lanes, values);
                        for (int64_t lane = 0; lane < 8; ++lane) {
                            panel_b[(k + lane) * kGemmAvx2NR + col] = lanes[lane];
                        }
                    }
                    for (; k < kc; ++k) {
                        const float value = WeightOps::LoadOne(source + static_cast<size_t>(k) * sizeof(uint16_t));
                        panel_b[k * kGemmAvx2NR + col] = value;
                        max_b = std::max(max_b, std::fabs(value));
                    }
                }
                alignas(32) float magnitudes[8];
                _mm256_store_ps(magnitudes, max_b_vector);
                for (const float value: magnitudes) {
                    max_b = std::max(max_b, value);
                }
                if (!GemmReductionMagnitudeIsSafe(max_a, max_b, args.k)) {
                    return false;
                }
                for (int64_t row = 0; row < rows; row += kGemmAvx2MR) {
                    AccumulateF32Tile4x16Avx2(panel_a + row * kGemmAvx2KC, panel_b, kc,
                                              args.output_m_stride, output + (mb + row) * args.output_m_stride + nb);
                }
            }
        }
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

template<class WeightOps>
bool TryGemmA32W16Avx2Fma(const GemmArgs& args) noexcept {
    if (args.m == 0 || args.n == 0 || args.k == 0 ||
        args.lhs_k_stride != 1 || args.rhs_k_stride != 1 || args.output_n_stride != 1) {
        return false;
    }
    return args.m <= 8 ? TryGemmW16Direct<WeightOps>(args) : TryGemmW16Blocked<WeightOps>(args);
}

} // namespace aethermind::cpu::detail
#endif
