#include "add_rmsnorm_internal.h"

#include <cmath>
#include <immintrin.h>

namespace aethermind::cpu::detail {
namespace {

void RunRow(const float* input, const float* residual, const float* weight,
            float* output, float* new_residual, int64_t hidden_size, float eps) noexcept {
    double sum_sq = 0.0;
    int64_t col = 0;
    for (; col <= hidden_size - 8; col += 8) {
        const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(input + col),
                                         _mm256_loadu_ps(residual + col));
        _mm256_storeu_ps(new_residual + col, sum);
        const __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(sum));
        const __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(sum, 1));
        double squares[8];
        _mm256_storeu_pd(squares, _mm256_mul_pd(lo, lo));
        _mm256_storeu_pd(squares + 4, _mm256_mul_pd(hi, hi));
        // Preserve the reference's ascending FP64 reduction after FP32 Add.
        for (double square: squares) {
            sum_sq += square;
        }
    }
    for (; col < hidden_size; ++col) {
        const float sum = input[col] + residual[col];
        new_residual[col] = sum;
        const double value = sum;
        sum_sq += value * value;
    }

    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(hidden_size) + eps);
    const __m256d scale = _mm256_set1_pd(inv_rms);
    col = 0;
    for (; col <= hidden_size - 4; col += 4) {
        const __m256d sum = _mm256_cvtps_pd(_mm_loadu_ps(new_residual + col));
        const __m256d w = _mm256_cvtps_pd(_mm_loadu_ps(weight + col));
        const __m256d norm = _mm256_mul_pd(_mm256_mul_pd(sum, scale), w);
        _mm_storeu_ps(output + col, _mm256_cvtpd_ps(norm));
    }
    for (; col < hidden_size; ++col) {
        output[col] = static_cast<float>(new_residual[col] * inv_rms * weight[col]);
    }
}

} // namespace

Status RunAddRmsNormF32Avx2(const AddRmsNormA32W32KernelArgs& args) noexcept {
    for (int64_t row = 0; row < args.row_count; ++row) {
        RunRow(args.input + row * args.input_row_stride,
               args.residual + row * args.residual_row_stride, args.weight,
               args.output + row * args.output_row_stride,
               args.new_residual + row * args.new_residual_row_stride, args.hidden_size, args.eps);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
