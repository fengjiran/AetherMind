#include "aethermind/backend/cpu/kernels/common/simd_utils.h"
#include "gemm_internal.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind::cpu::detail {
namespace {

#if defined(__AVX2__) && defined(__FMA__)

constexpr int64_t kAvx2Lanes = 8;
constexpr int64_t kAvx2KUnroll = 2;
constexpr int64_t kAvx2KBlock = kAvx2Lanes * kAvx2KUnroll;
constexpr int64_t kAvx2OutputBlock = 4;

float RunM1KContiguousAvx2Dot(const float* lhs, const float* weight, int64_t k) noexcept {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int64_t inner = 0;
    for (; inner + kAvx2KBlock <= k; inner += kAvx2KBlock) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + inner),
                               _mm256_loadu_ps(weight + inner),
                               sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + inner + kAvx2Lanes),
                               _mm256_loadu_ps(weight + inner + kAvx2Lanes),
                               sum1);
    }

    float sum = HorizontalSumAvx2(_mm256_add_ps(sum0, sum1));
    for (; inner < k; ++inner) {
        sum += lhs[inner] * weight[inner];
    }
    return sum;
}

void RunM1KContiguousAvx2Fma(const GemmF32Args& args) noexcept {
    int64_t col = 0;
    for (; col + kAvx2OutputBlock <= args.n; col += kAvx2OutputBlock) {
        const float* weight0 = args.rhs + col * args.rhs_n_stride;
        const float* weight1 = weight0 + args.rhs_n_stride;
        const float* weight2 = weight1 + args.rhs_n_stride;
        const float* weight3 = weight2 + args.rhs_n_stride;
        __m256 sum00 = _mm256_setzero_ps();
        __m256 sum01 = _mm256_setzero_ps();
        __m256 sum10 = _mm256_setzero_ps();
        __m256 sum11 = _mm256_setzero_ps();
        __m256 sum20 = _mm256_setzero_ps();
        __m256 sum21 = _mm256_setzero_ps();
        __m256 sum30 = _mm256_setzero_ps();
        __m256 sum31 = _mm256_setzero_ps();

        int64_t inner = 0;
        for (; inner + kAvx2KBlock <= args.k; inner += kAvx2KBlock) {
            const __m256 lhs0 = _mm256_loadu_ps(args.lhs + inner);
            const __m256 lhs1 = _mm256_loadu_ps(args.lhs + inner + kAvx2Lanes);
            sum00 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(weight0 + inner), sum00);
            sum01 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(weight0 + inner + kAvx2Lanes), sum01);
            sum10 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(weight1 + inner), sum10);
            sum11 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(weight1 + inner + kAvx2Lanes), sum11);
            sum20 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(weight2 + inner), sum20);
            sum21 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(weight2 + inner + kAvx2Lanes), sum21);
            sum30 = _mm256_fmadd_ps(lhs0, _mm256_loadu_ps(weight3 + inner), sum30);
            sum31 = _mm256_fmadd_ps(lhs1, _mm256_loadu_ps(weight3 + inner + kAvx2Lanes), sum31);
        }

        float result0 = HorizontalSumAvx2(_mm256_add_ps(sum00, sum01));
        float result1 = HorizontalSumAvx2(_mm256_add_ps(sum10, sum11));
        float result2 = HorizontalSumAvx2(_mm256_add_ps(sum20, sum21));
        float result3 = HorizontalSumAvx2(_mm256_add_ps(sum30, sum31));
        for (; inner < args.k; ++inner) {
            const float input = args.lhs[inner];
            result0 += input * weight0[inner];
            result1 += input * weight1[inner];
            result2 += input * weight2[inner];
            result3 += input * weight3[inner];
        }

        args.output[col] = result0;
        args.output[col + 1] = result1;
        args.output[col + 2] = result2;
        args.output[col + 3] = result3;
    }

    for (; col < args.n; ++col) {
        args.output[col] = RunM1KContiguousAvx2Dot(
                args.lhs, args.rhs + col * args.rhs_n_stride, args.k);
    }
}

#endif

} // namespace

#if defined(__AVX2__) && defined(__FMA__)
Status RunGemmF32Avx2Fma(const GemmF32Args& args) noexcept {
    if (args.m == 1 && args.k > 0 && args.lhs_k_stride == 1 &&
        args.rhs_k_stride == 1 && args.output_n_stride == 1) {
        RunM1KContiguousAvx2Fma(args);
        return Status::Ok();
    }
    return RunGemmF32Scalar(args);
}
#endif

} // namespace aethermind::cpu::detail
