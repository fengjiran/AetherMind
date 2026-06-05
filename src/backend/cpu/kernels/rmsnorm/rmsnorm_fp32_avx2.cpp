#include "aethermind/backend/cpu/kernels/common/cpu_simd_utils.h"
#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "rmsnorm_internal.h"

#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace aethermind {
namespace {

const CpuRmsNormParams* GetParams(const void* packed_params) noexcept {
    return static_cast<const CpuRmsNormParams*>(packed_params);
}

#if !defined(__AVX2__) || !defined(__FMA__)
void ProcessStridedRmsNormRowScalar(const RmsNormFp32KernelArgs& args, int64_t row_idx) noexcept {
    const float* const row_in = args.input + row_idx * args.input_row_stride;
    float* const row_out = args.output + row_idx * args.output_row_stride;

    double sum_sq = 0.0;
    for (int64_t j = 0; j < args.hidden_size; ++j) {
        const auto x = static_cast<double>(row_in[j * args.input_col_stride]);
        sum_sq += x * x;
    }

    const double mean_sq = sum_sq / static_cast<double>(args.hidden_size);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(args.epsilon));
    for (int64_t j = 0; j < args.hidden_size; ++j) {
        row_out[j * args.output_col_stride] = static_cast<float>(static_cast<double>(row_in[j * args.input_col_stride]) *
                                                                 inv_rms * static_cast<double>(args.weight[j * args.weight_stride]));
    }
}
#endif

}// namespace

#if defined(__AVX2__) && defined(__FMA__)
AM_ALWAYS_INLINE void rmsnorm_micro_kernel_avx2(float* __restrict__ output,
                                                const float* __restrict__ input,
                                                const float* __restrict__ weight,
                                                int64_t hidden_size,
                                                float epsilon) {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    int64_t j = 0;
    for (; j + 32 <= hidden_size; j += 32) {
        const __m256 x0 = _mm256_loadu_ps(input + j);
        const __m256 x1 = _mm256_loadu_ps(input + j + 8);
        const __m256 x2 = _mm256_loadu_ps(input + j + 16);
        const __m256 x3 = _mm256_loadu_ps(input + j + 24);

        vsum0 = _mm256_fmadd_ps(x0, x0, vsum0);
        vsum1 = _mm256_fmadd_ps(x1, x1, vsum1);
        vsum2 = _mm256_fmadd_ps(x2, x2, vsum2);
        vsum3 = _mm256_fmadd_ps(x3, x3, vsum3);
    }

    __m256 vres = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1), _mm256_add_ps(vsum2, vsum3));
    for (; j + 8 <= hidden_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(input + j);
        vres = _mm256_fmadd_ps(x0, x0, vres);
    }

    float sum_sq = HorizontalSumAvx2(vres);
    for (; j < hidden_size; ++j) {
        sum_sq += input[j] * input[j];
    }

    const float mean_sq = sum_sq / static_cast<float>(hidden_size);
    const float inv_rms = 1.0F / std::sqrt(mean_sq + epsilon);
    const __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

    j = 0;
    for (; j + 32 <= hidden_size; j += 32) {
        __m256 x0 = _mm256_loadu_ps(input + j);
        __m256 x1 = _mm256_loadu_ps(input + j + 8);
        __m256 x2 = _mm256_loadu_ps(input + j + 16);
        __m256 x3 = _mm256_loadu_ps(input + j + 24);

        x0 = _mm256_mul_ps(x0, inv_rms_vec);
        x1 = _mm256_mul_ps(x1, inv_rms_vec);
        x2 = _mm256_mul_ps(x2, inv_rms_vec);
        x3 = _mm256_mul_ps(x3, inv_rms_vec);

        const __m256 w0 = _mm256_loadu_ps(weight + j);
        const __m256 w1 = _mm256_loadu_ps(weight + j + 8);
        const __m256 w2 = _mm256_loadu_ps(weight + j + 16);
        const __m256 w3 = _mm256_loadu_ps(weight + j + 24);

        const __m256 out0 = _mm256_mul_ps(x0, w0);
        const __m256 out1 = _mm256_mul_ps(x1, w1);
        const __m256 out2 = _mm256_mul_ps(x2, w2);
        const __m256 out3 = _mm256_mul_ps(x3, w3);

        _mm256_storeu_ps(output + j, out0);
        _mm256_storeu_ps(output + j + 8, out1);
        _mm256_storeu_ps(output + j + 16, out2);
        _mm256_storeu_ps(output + j + 24, out3);
    }

    for (; j + 8 <= hidden_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(input + j);
        const __m256 w0 = _mm256_loadu_ps(weight + j);
        x0 = _mm256_mul_ps(x0, inv_rms_vec);
        _mm256_storeu_ps(output + j, _mm256_mul_ps(x0, w0));
    }

    for (; j < hidden_size; ++j) {
        output[j] = input[j] * inv_rms * weight[j];
    }
}
#endif

/// Executes RMSNorm on already-validated low-level arguments.
///
/// Callers must guarantee non-null data pointers, positive dimensions, positive
/// strides, unit column strides (input_col_stride_, weight_stride_,
/// output_col_stride_ all equal 1), finite positive epsilon, and sufficient
/// backing storage for every addressed element. Runtime validation belongs in
/// CpuRmsNormKernelEntry.
Status RmsNormKernel_CPU_FP32_AVX2(const RmsNormFp32KernelArgs& args) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    auto process_row = [&args](int64_t row_idx) noexcept {
        rmsnorm_micro_kernel_avx2(args.output + row_idx * args.output_row_stride,
                                  args.input + row_idx * args.input_row_stride,
                                  args.weight,
                                  args.hidden_size,
                                  args.epsilon);
    };
#else
    auto process_row = [&args](int64_t row_idx) noexcept {
        ProcessStridedRmsNormRowScalar(args, row_idx);
    };
#endif

    if (constexpr int64_t kOmpParallelThreshold = 16; args.seq_len <= kOmpParallelThreshold) {
        for (int64_t i = 0; i < args.seq_len; ++i) {
            process_row(i);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < args.seq_len; ++i) {
            process_row(i);
        }
    }

    return Status::Ok();
}

}// namespace aethermind
