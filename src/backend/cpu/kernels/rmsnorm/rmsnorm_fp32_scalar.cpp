#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_static_registration.h"
#include "rmsnorm_internal.h"

#include <cmath>

namespace aethermind {
namespace {

AM_ALWAYS_INLINE void micro_kernel_fp32_scalar(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  const float* __restrict__ weight,
                                  int64_t hidden_size,
                                  int64_t input_stride,
                                  int64_t weight_stride,
                                  int64_t output_stride,
                                  float epsilon) {
    double sum_sq = 0.0;
    for (int64_t j = 0; j < hidden_size; ++j) {
        const auto x = static_cast<double>(input[j * input_stride]);
        sum_sq += x * x;
    }

    const double mean_sq = sum_sq / static_cast<double>(hidden_size);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(epsilon));
    for (int64_t j = 0; j < hidden_size; ++j) {
        const auto x = static_cast<double>(input[j * input_stride]);
        const auto w = static_cast<double>(weight[j * weight_stride]);
        output[j * output_stride] = static_cast<float>(x * inv_rms * w);
    }
}

}// namespace


Status RmsNormKernel_CPU_FP32_Scalar(const RmsNormFp32KernelArgs& args) noexcept {
    if (constexpr int64_t kOmpParallelThreshold = 16; args.seq_len <= kOmpParallelThreshold) {
        for (int64_t i = 0; i < args.seq_len; ++i) {
            micro_kernel_fp32_scalar(args.output + i * args.output_row_stride,
                        args.input + i * args.input_row_stride,
                        args.weight,
                        args.hidden_size,
                        args.input_col_stride,
                        args.weight_stride,
                        args.output_col_stride,
                        args.eps);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < args.seq_len; ++i) {
            micro_kernel_fp32_scalar(args.output + i * args.output_row_stride,
                        args.input + i * args.input_row_stride,
                        args.weight,
                        args.hidden_size,
                        args.input_col_stride,
                        args.weight_stride,
                        args.output_col_stride,
                        args.eps);
        }
    }

    return Status::Ok();
}

}// namespace aethermind
