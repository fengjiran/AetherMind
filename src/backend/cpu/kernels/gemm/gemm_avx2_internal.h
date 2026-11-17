#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_AVX2_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_GEMM_AVX2_INTERNAL_H

#include <cstdint>
#include <immintrin.h>

namespace aethermind::cpu::detail {

inline constexpr int64_t kGemmAvx2KC = 512;
inline constexpr int64_t kGemmAvx2NR = 16;
inline constexpr int64_t kGemmAvx2MR = 4;
inline constexpr int64_t kGemmAvx2MB = 48;

// Shared F32 accumulation tile. Callers own precision-specific panel loading,
// shape/stride admission, tails and numerical fallbacks.
inline void AccumulateF32Tile4x16Avx2(const float* buf_a, const float* buf_b,
                                      int64_t kc_len, int64_t output_m_stride,
                                      float* out_row0) noexcept {
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps();
    __m256 acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps();
    __m256 acc31 = _mm256_setzero_ps();

    const float* a0 = buf_a;
    const float* a1 = a0 + kGemmAvx2KC;
    const float* a2 = a1 + kGemmAvx2KC;
    const float* a3 = a2 + kGemmAvx2KC;

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

} // namespace aethermind::cpu::detail
#endif
