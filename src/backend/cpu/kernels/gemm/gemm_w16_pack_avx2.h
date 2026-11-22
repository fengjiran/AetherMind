#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_W16_PACK_AVX2_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_W16_PACK_AVX2_H

#include "aethermind/dtypes/half.h"
#include "gemm_avx2_internal.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace aethermind::cpu::detail {

struct Bf16GemmWeightOps {
    static constexpr size_t kElementBytes = sizeof(uint16_t);

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
    static constexpr size_t kElementBytes = sizeof(uint16_t);

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
float PackGemmW16BPanelAvx2(const std::byte* weight, size_t row_bytes,
                            int64_t kc, float* panel) noexcept {
    return PackGemmBPanelAvx2<WeightOps, true>(weight, row_bytes, kc, panel);
}

} // namespace aethermind::cpu::detail
#endif
