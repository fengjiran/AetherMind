#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_W16_PACK_AVX2_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_W16_PACK_AVX2_H

#include "aethermind/dtypes/half.h"
#include "gemm_avx2_internal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace aethermind::cpu::detail {

struct Bf16GemmWeightOps {
    static __m256 LoadEight(const std::byte* data) noexcept {
        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }

    static float LoadOne(const std::byte* data) noexcept {
        uint16_t bits;
        std::memcpy(&bits, data, sizeof(bits));
        return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16U);
    }
};

#if defined(__F16C__)
struct F16GemmWeightOps {
    static __m256 LoadEight(const std::byte* data) noexcept {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data)));
    }

    static float LoadOne(const std::byte* data) noexcept {
        uint16_t bits;
        std::memcpy(&bits, data, sizeof(bits));
        return Half(bits, Half::from_bits());
    }
};
#endif

inline bool GemmReductionMagnitudeIsSafe(float input, float weight, int64_t k) noexcept {
    // Bounding the sum of absolute products also bounds every scalar/SIMD
    // partial sum. Extreme inputs retain the reference's overflow semantics.
    return std::isfinite(input) && std::isfinite(weight) &&
           static_cast<double>(input) * static_cast<double>(weight) * static_cast<double>(k) <=
                   static_cast<double>(std::numeric_limits<float>::max()) * 0.5;
}

template<class WeightOps>
float PackGemmW16BPanelAvx2(const std::byte* weight, size_t row_bytes, int64_t kc, float* panel) noexcept {
    static_assert(kGemmAvx2NR % 8 == 0);
    __m256 max_vector = _mm256_setzero_ps();
    float max_tail = 0.0F;
    // Pack eight weight rows at a time to limit active source lines when
    // power-of-two row strides map the rows to the same L1 cache sets.
    for (int64_t col = 0; col < kGemmAvx2NR; col += 8) {
        const auto* stripe = weight + static_cast<size_t>(col) * row_bytes;
        int64_t k = 0;
        for (; k + 8 <= kc; k += 8) {
            const auto* source = stripe + static_cast<size_t>(k) * sizeof(uint16_t);
            __m256 rows[8];
            for (size_t lane = 0; lane < 8; ++lane) {
                rows[lane] = WeightOps::LoadEight(source + lane * row_bytes);
                const __m256 magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0F), rows[lane]);
                // Keep the finite maximum even when a weight is NaN. The
                // caller detects non-finite results and recomputes the output.
                max_vector = _mm256_max_ps(magnitude, max_vector);
            }
            StoreGemmBTile8x8Avx2(rows, panel + k * kGemmAvx2NR + col);
        }
        for (; k < kc; ++k) {
            const auto* source = stripe + static_cast<size_t>(k) * sizeof(uint16_t);
            for (size_t lane = 0; lane < 8; ++lane) {
                const float value = WeightOps::LoadOne(source + lane * row_bytes);
                panel[k * kGemmAvx2NR + col + lane] = value;
                max_tail = std::max(max_tail, std::fabs(value));
            }
        }
    }
    alignas(32) float magnitudes[8];
    _mm256_store_ps(magnitudes, max_vector);
    for (const float value: magnitudes) {
        max_tail = std::max(max_tail, value);
    }
    return max_tail;
}

} // namespace aethermind::cpu::detail
#endif
