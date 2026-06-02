#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_REFERENCE_H

#include <cmath>

namespace aethermind {

struct ReferenceRmsNormArgs {
    const float* input_{};
    const float* weight_{};
    float* output_{};
    int64_t seq_len_{};
    int64_t hidden_size_{};
    float epsilon_{1.0e-5F};
};

inline void ReferenceRmsNorm(const ReferenceRmsNormArgs& args) noexcept {
    for (int64_t s = 0; s < args.seq_len_; ++s) {
        const float* row_in = args.input_ + s * args.hidden_size_;
        float* row_out = args.output_ + s * args.hidden_size_;

        double mean_square = 0.0;
        for (int64_t i = 0; i < args.hidden_size_; ++i) {
            const auto x = static_cast<double>(row_in[i]);
            mean_square += x * x;
        }
        mean_square /= static_cast<double>(args.hidden_size_);

        const double inv_rms = 1.0 / std::sqrt(mean_square + static_cast<double>(args.epsilon_));
        for (int64_t i = 0; i < args.hidden_size_; ++i) {
            row_out[i] = static_cast<float>(static_cast<double>(row_in[i]) * inv_rms * static_cast<double>(args.weight_[i]));
        }
    }
}

}// namespace aethermind

#endif
