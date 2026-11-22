#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_AVX2_INTERNAL_H

/// @file gemm_avx2_internal.h
/// @brief Shared AVX2 F32 GEMM tiles, B-panel packing and blocking constants.
///
/// Only translation units compiled with AVX2 and FMA enabled (`-mavx2 -mfma`)
/// may include this header: it exposes `<immintrin.h>` types and intrinsics
/// whose inline definitions are compiled into every consuming TU.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace aethermind::cpu::detail {

inline constexpr int64_t kGemmAvx2KC = 512;
inline constexpr int64_t kGemmAvx2NR = 16;
inline constexpr int64_t kGemmAvx2MR = 4;
inline constexpr int64_t kGemmAvx2MB = 48;

// NOLINTBEGIN(portability-simd-intrinsics)
// This microkernel is intentionally AVX2-specific; per-ISA kernels are the
// backend's design, so the portability check does not apply.
// A source row holds eight consecutive K values from one weight row. Write
// each transposed row to eight adjacent N lanes of the F32 compute panel.
inline void StoreGemmBTile8x8Avx2(const __m256 (&rows)[8], float* out) noexcept {
    const __m256 pair01_lo = _mm256_unpacklo_ps(rows[0], rows[1]);
    const __m256 pair01_hi = _mm256_unpackhi_ps(rows[0], rows[1]);
    const __m256 pair23_lo = _mm256_unpacklo_ps(rows[2], rows[3]);
    const __m256 pair23_hi = _mm256_unpackhi_ps(rows[2], rows[3]);
    const __m256 pair45_lo = _mm256_unpacklo_ps(rows[4], rows[5]);
    const __m256 pair45_hi = _mm256_unpackhi_ps(rows[4], rows[5]);
    const __m256 pair67_lo = _mm256_unpacklo_ps(rows[6], rows[7]);
    const __m256 pair67_hi = _mm256_unpackhi_ps(rows[6], rows[7]);

    const __m256 first4[4]{
            _mm256_shuffle_ps(pair01_lo, pair23_lo, 0x44),
            _mm256_shuffle_ps(pair01_lo, pair23_lo, 0xee),
            _mm256_shuffle_ps(pair01_hi, pair23_hi, 0x44),
            _mm256_shuffle_ps(pair01_hi, pair23_hi, 0xee)};
    const __m256 last4[4]{
            _mm256_shuffle_ps(pair45_lo, pair67_lo, 0x44),
            _mm256_shuffle_ps(pair45_lo, pair67_lo, 0xee),
            _mm256_shuffle_ps(pair45_hi, pair67_hi, 0x44),
            _mm256_shuffle_ps(pair45_hi, pair67_hi, 0xee)};
    for (int i = 0; i < 4; ++i) {
        _mm256_store_ps(out + i * kGemmAvx2NR, _mm256_permute2f128_ps(first4[i], last4[i], 0x20));
        _mm256_store_ps(out + (i + 4) * kGemmAvx2NR, _mm256_permute2f128_ps(first4[i], last4[i], 0x31));
    }
}

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

// Packs NR full B rows (row stride in bytes) into the k-major panel layout
// used by AccumulateF32Tile4x16Avx2: full 8x8 K blocks are transposed with
// StoreGemmBTile8x8Avx2, the K remainder is scalar. With TrackMagnitude the
// largest finite |weight| is returned; otherwise the return value is 0.
template<class WeightOps, bool TrackMagnitude>
float PackGemmBPanelAvx2(const std::byte* weight, size_t row_bytes,
                         int64_t kc, float* panel) noexcept {
    static_assert(kGemmAvx2NR % 8 == 0);
    __m256 max_vector = _mm256_setzero_ps();
    float max_tail = 0.0F;
    // Pack eight weight rows at a time to limit active source lines when
    // power-of-two row strides map the rows to the same L1 cache sets.
    for (int64_t col = 0; col < kGemmAvx2NR; col += 8) {
        const auto* stripe = weight + static_cast<size_t>(col) * row_bytes;
        int64_t k = 0;
        for (; k + 8 <= kc; k += 8) {
            const auto* source = stripe + static_cast<size_t>(k) * WeightOps::kElementBytes;
            __m256 rows[8];
            for (size_t lane = 0; lane < 8; ++lane) {
                rows[lane] = WeightOps::LoadEight(source + lane * row_bytes);
                if constexpr (TrackMagnitude) {
                    const __m256 magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0F), rows[lane]);
                    // Keep the finite maximum even when a weight is NaN. The
                    // caller detects non-finite results and recomputes the output.
                    max_vector = _mm256_max_ps(magnitude, max_vector);
                }
            }
            StoreGemmBTile8x8Avx2(rows, panel + k * kGemmAvx2NR + col);
        }

        for (; k < kc; ++k) {
            const auto* source = stripe + static_cast<size_t>(k) * WeightOps::kElementBytes;
            for (size_t lane = 0; lane < 8; ++lane) {
                const float value = WeightOps::LoadOne(source + lane * row_bytes);
                panel[k * kGemmAvx2NR + col + lane] = value;
                if constexpr (TrackMagnitude) {
                    max_tail = std::max(max_tail, std::fabs(value));
                }
            }
        }
    }

    if constexpr (TrackMagnitude) {
        alignas(32) float magnitudes[8];
        _mm256_store_ps(magnitudes, max_vector);
        for (const float value: magnitudes) {
            max_tail = std::max(max_tail, value);
        }
    }
    return max_tail;
}

// Shared F32 accumulation tile. Callers own precision-specific panel loading,
// shape/stride admission, tails and numerical fallbacks.
inline void AccumulateF32Tile4x16Avx2(const float* buf_a, const float* buf_b,
                                      int64_t kc_len, int64_t output_m_stride,
                                      float* out_row0,
                                      int64_t a_row_stride = kGemmAvx2KC) noexcept {
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps();
    __m256 acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps();
    __m256 acc31 = _mm256_setzero_ps();

    const float* a0 = buf_a;
    const float* a1 = a0 + a_row_stride;
    const float* a2 = a1 + a_row_stride;
    const float* a3 = a2 + a_row_stride;

    for (int64_t kk = 0; kk < kc_len; ++kk) {
        const __m256 x0 = _mm256_set1_ps(a0[kk]);
        const __m256 x1 = _mm256_set1_ps(a1[kk]);
        const __m256 x2 = _mm256_set1_ps(a2[kk]);
        const __m256 x3 = _mm256_set1_ps(a3[kk]);
        const __m256 b0 = _mm256_loadu_ps(buf_b + kk * kGemmAvx2NR);
        const __m256 b1 = _mm256_loadu_ps(buf_b + kk * kGemmAvx2NR + 8);
        acc00 = _mm256_fmadd_ps(x0, b0, acc00);
        acc01 = _mm256_fmadd_ps(x0, b1, acc01);
        acc10 = _mm256_fmadd_ps(x1, b0, acc10);
        acc11 = _mm256_fmadd_ps(x1, b1, acc11);
        acc20 = _mm256_fmadd_ps(x2, b0, acc20);
        acc21 = _mm256_fmadd_ps(x2, b1, acc21);
        acc30 = _mm256_fmadd_ps(x3, b0, acc30);
        acc31 = _mm256_fmadd_ps(x3, b1, acc31);
    }

    float* out0 = out_row0;
    float* out1 = out0 + output_m_stride;
    float* out2 = out1 + output_m_stride;
    float* out3 = out2 + output_m_stride;
    __m256 o00 = _mm256_loadu_ps(out0);
    __m256 o01 = _mm256_loadu_ps(out0 + 8);
    __m256 o10 = _mm256_loadu_ps(out1);
    __m256 o11 = _mm256_loadu_ps(out1 + 8);
    __m256 o20 = _mm256_loadu_ps(out2);
    __m256 o21 = _mm256_loadu_ps(out2 + 8);
    __m256 o30 = _mm256_loadu_ps(out3);
    __m256 o31 = _mm256_loadu_ps(out3 + 8);
    _mm256_storeu_ps(out0, _mm256_add_ps(o00, acc00));
    _mm256_storeu_ps(out0 + 8, _mm256_add_ps(o01, acc01));
    _mm256_storeu_ps(out1, _mm256_add_ps(o10, acc10));
    _mm256_storeu_ps(out1 + 8, _mm256_add_ps(o11, acc11));
    _mm256_storeu_ps(out2, _mm256_add_ps(o20, acc20));
    _mm256_storeu_ps(out2 + 8, _mm256_add_ps(o21, acc21));
    _mm256_storeu_ps(out3, _mm256_add_ps(o30, acc30));
    _mm256_storeu_ps(out3 + 8, _mm256_add_ps(o31, acc31));
}
// NOLINTEND(portability-simd-intrinsics)

} // namespace aethermind::cpu::detail
#endif
