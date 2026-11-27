#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_AVX2_INTERNAL_H

/// @file gemm_avx2_internal.h
/// @brief Shared dtype-agnostic AVX2 GEMM constants, weight adapters, B-panel
///        packing and F32 tile kernels.
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
inline constexpr int64_t kGemmAvx2MR6 = 6;
inline constexpr int64_t kGemmAvx2MB = 48;

/// @brief Weight adapter for plain F32 rows consumed by PackGemmBPanelAvx2.
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

// Packs NR full B rows (row stride in bytes) into the k-major panel layout
// used by the F32 tile kernels: full 8x8 K blocks are transposed with
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

// Shared Nx16 F32 accumulation core (N in {1, 2, 4, 6}): two independent 8-wide
// accumulator chains per row span the 16 output columns, and the K loop is
// unrolled by 4 to amortize loop overhead. With N=6 the twelve chains keep
// both FMA ports busy; the smaller tiles cover M-tail rows with the same
// panel flow instead of falling back to the row-pair kernel. The B panel is
// k-major with an NR=16 row stride; `load_a(r, kk)` supplies the A operand
// for row r at K index kk and differs between the direct-broadcast and
// broadcast-packed variants. `accumulate=false` overwrites the output tile and
// is used by the first K panel; later panels add their partial sums.
template<int Rows, typename LoadA>
void AccumulateF32TileNx16CoreAvx2(const float* buf_b, int64_t kc_len,
                                   int64_t output_m_stride, float* out_row0,
                                   LoadA load_a, bool accumulate) noexcept {
    static_assert(Rows >= 1 && Rows <= 6);
    __m256 acc[Rows][2];
    for (int r = 0; r < Rows; ++r) {
        acc[r][0] = _mm256_setzero_ps();
        acc[r][1] = _mm256_setzero_ps();
    }

    int64_t kk = 0;
    for (; kk + 4 <= kc_len; kk += 4) {
        for (int s = 0; s < 4; ++s) {
            const __m256 b0 = _mm256_load_ps(buf_b + (kk + s) * kGemmAvx2NR);
            const __m256 b1 = _mm256_load_ps(buf_b + (kk + s) * kGemmAvx2NR + 8);
            for (int r = 0; r < Rows; ++r) {
                const __m256 x = load_a(r, kk + s);
                acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
            }
        }
    }

    for (; kk < kc_len; ++kk) {
        const __m256 b0 = _mm256_load_ps(buf_b + kk * kGemmAvx2NR);
        const __m256 b1 = _mm256_load_ps(buf_b + kk * kGemmAvx2NR + 8);
        for (int r = 0; r < Rows; ++r) {
            const __m256 x = load_a(r, kk);
            acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
        }
    }

    for (int r = 0; r < Rows; ++r) {
        float* const out = out_row0 + r * output_m_stride;
        __m256 o0 = acc[r][0];
        __m256 o1 = acc[r][1];
        if (accumulate) {
            o0 = _mm256_add_ps(_mm256_loadu_ps(out), o0);
            o1 = _mm256_add_ps(_mm256_loadu_ps(out + 8), o1);
        }
        _mm256_storeu_ps(out, o0);
        _mm256_storeu_ps(out + 8, o1);
    }
}

// Nx16 tiles reading A directly from the row-major lhs: every A operand is a
// scalar broadcast from a strided row, no A packing cost and no layout change.
inline void AccumulateF32Tile6x16Avx2(const float* buf_a, const float* buf_b,
                                      int64_t kc_len, int64_t output_m_stride,
                                      float* out_row0, int64_t a_row_stride,
                                      bool accumulate = true) noexcept {
    AccumulateF32TileNx16CoreAvx2<6>(
            buf_b, kc_len, output_m_stride, out_row0,
            [=](int r, int64_t k) {
                return _mm256_broadcast_ss(buf_a + r * a_row_stride + k);
            },
            accumulate);
}

inline void AccumulateF32Tile4x16Avx2(const float* buf_a, const float* buf_b,
                                      int64_t kc_len, int64_t output_m_stride,
                                      float* out_row0, int64_t a_row_stride,
                                      bool accumulate = true) noexcept {
    AccumulateF32TileNx16CoreAvx2<4>(
            buf_b, kc_len, output_m_stride, out_row0,
            [=](int r, int64_t k) {
                return _mm256_broadcast_ss(buf_a + r * a_row_stride + k);
            },
            accumulate);
}

inline void AccumulateF32Tile2x16Avx2(const float* buf_a, const float* buf_b,
                                      int64_t kc_len, int64_t output_m_stride,
                                      float* out_row0, int64_t a_row_stride,
                                      bool accumulate = true) noexcept {
    AccumulateF32TileNx16CoreAvx2<2>(
            buf_b, kc_len, output_m_stride, out_row0,
            [=](int r, int64_t k) {
                return _mm256_broadcast_ss(buf_a + r * a_row_stride + k);
            },
            accumulate);
}

// NOLINTEND(portability-simd-intrinsics)

} // namespace aethermind::cpu::detail
#endif
