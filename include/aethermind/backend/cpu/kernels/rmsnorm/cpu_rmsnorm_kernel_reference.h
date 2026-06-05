#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H

#include "cpu_rmsnorm_kernel.h"

namespace aethermind {

inline void ReferenceRmsNorm(const RmsNormArgs& args) noexcept {
    const auto* const input = static_cast<const float*>(args.input);
    const auto* const weight = static_cast<const float*>(args.weight);
    auto* const output = static_cast<float*>(args.output);

    for (int64_t s = 0; s < args.seq_len; ++s) {
        const float* const row_in = input + s * args.input_row_stride;
        float* const row_out = output + s * args.output_row_stride;

        double sum_sq = 0.0;
        for (int64_t j = 0; j < args.hidden_size; ++j) {
            const auto x = static_cast<double>(row_in[j * args.input_col_stride]);
            sum_sq += x * x;
        }

        const double mean_sq = sum_sq / static_cast<double>(args.hidden_size);
        const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(args.epsilon));
        for (int64_t j = 0; j < args.hidden_size; ++j) {
            row_out[j * args.output_col_stride] = static_cast<float>(
                    static_cast<double>(row_in[j * args.input_col_stride]) * inv_rms *
                    static_cast<double>(weight[j * args.weight_stride]));
        }
    }
}

}// namespace aethermind

#endif
