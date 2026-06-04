#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H

#include "cpu_rmsnorm_kernel.h"

namespace aethermind {

inline void ReferenceRmsNorm(const CpuRmsNormKernelArgs& args) noexcept {
    for (int64_t s = 0; s < args.seq_len_; ++s) {
        const float* const row_in = args.input_ + s * args.input_row_stride_;
        float* const row_out = args.output_ + s * args.output_row_stride_;

        double sum_sq = 0.0;
        for (int64_t j = 0; j < args.hidden_size_; ++j) {
            const auto x = static_cast<double>(row_in[j * args.input_col_stride_]);
            sum_sq += x * x;
        }

        const double mean_sq = sum_sq / static_cast<double>(args.hidden_size_);
        const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(args.epsilon_));
        for (int64_t j = 0; j < args.hidden_size_; ++j) {
            row_out[j * args.output_col_stride_] = static_cast<float>(
                    static_cast<double>(row_in[j * args.input_col_stride_]) * inv_rms *
                    static_cast<double>(args.weight_[j * args.weight_stride_]));
        }
    }
}

}// namespace aethermind

#endif
