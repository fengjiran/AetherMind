#include "rmsnorm_internal.h"

#include <cmath>

namespace aethermind::cpu::detail {
namespace {

AM_ALWAYS_INLINE void RmsNormF32RowReference(float* output,
                                             const float* input,
                                             const float* weight,
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

} // namespace


Status RunRmsNormF32Reference(const RmsNormF32KernelArgs& args) noexcept {
    for (int64_t row = 0; row < args.row_count; ++row) {
        RmsNormF32RowReference(args.output + row * args.output_row_stride,
                               args.input + row * args.input_row_stride,
                               args.weight,
                               args.hidden_size,
                               args.input_col_stride,
                               args.weight_stride,
                               args.output_col_stride,
                               args.eps);
    }

    return Status::Ok();
}

} // namespace aethermind::cpu::detail
