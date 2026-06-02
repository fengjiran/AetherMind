#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H

#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"

#include <cmath>

namespace aethermind {

inline void ReferenceRmsNorm(const CpuRmsNormKernelArgs& args) noexcept {
    for (int64_t s = 0; s < args.seq_len_; ++s) {
        const float* row_in = args.input_ + s * args.input_row_stride_;
        float* row_out = args.output_ + s * args.output_row_stride_;

        double mean_square = 0.0;
        for (int64_t i = 0; i < args.hidden_size_; ++i) {
            const auto x = static_cast<double>(row_in[i * args.input_col_stride_]);
            mean_square += x * x;
        }
        mean_square /= static_cast<double>(args.hidden_size_);

        const double inv_rms = 1.0 / std::sqrt(mean_square + static_cast<double>(args.epsilon_));
        for (int64_t i = 0; i < args.hidden_size_; ++i) {
            row_out[i * args.output_col_stride_] = static_cast<float>(static_cast<double>(row_in[i * args.input_col_stride_]) * inv_rms * static_cast<double>(args.weight_[i * args.weight_stride_]));
        }
    }
}

}// namespace aethermind

#endif
