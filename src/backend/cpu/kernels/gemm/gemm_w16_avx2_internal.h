#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_W16_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_W16_AVX2_INTERNAL_H

/// @file gemm_w16_avx2_internal.h
/// @brief Shared AVX2 F16/BF16 weight-load policies and W16 B-panel packing
///        used by the A16W16 and A32W16 families.
///
/// Only translation units compiled with AVX2 and FMA enabled (`-mavx2 -mfma`)
/// may include this header; F16GemmWeightOps additionally requires F16C.

#include "aethermind/dtypes/half.h"
#include "gemm_avx2_internal.h"

#include <bit>
#include <cstddef>
#include <cstring>

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

template<class WeightOps>
float PackGemmW16BPanelAvx2(const std::byte* weight, size_t row_bytes,
                            int64_t kc, float* panel) noexcept {
    return PackGemmBPanelAvx2<WeightOps, true>(weight, row_bytes, kc, panel);
}

} // namespace aethermind::cpu::detail
#endif