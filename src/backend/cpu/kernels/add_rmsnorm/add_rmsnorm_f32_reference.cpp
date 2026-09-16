#include "add_rmsnorm_internal.h"

#include <cmath>

namespace aethermind::cpu::detail {
namespace {

void RunAddRmsNormF32RowReference(const AddRmsNormF32KernelArgs& args,
                                  int64_t row) noexcept {
    const float* const input_row = args.input + row * args.input_row_stride;
    const float* const residual_row = args.residual + row * args.residual_row_stride;
    float* const output_row = args.output + row * args.output_row_stride;
    float* const new_residual_row = args.new_residual + row * args.new_residual_row_stride;

    double sum_sq = 0.0;
    for (int64_t col = 0; col < args.hidden_size; ++col) {
        const float sum = input_row[col * args.input_col_stride] +
                          residual_row[col * args.residual_col_stride];
        new_residual_row[col * args.new_residual_col_stride] = sum;

        const auto value = static_cast<double>(sum);
        sum_sq += value * value;
    }

    const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(args.hidden_size) +
                                           static_cast<double>(args.eps));
    for (int64_t col = 0; col < args.hidden_size; ++col) {
        const auto sum = static_cast<double>(new_residual_row[col * args.new_residual_col_stride]);
        const auto weight = static_cast<double>(args.weight[col * args.weight_stride]);
        output_row[col * args.output_col_stride] = static_cast<float>(sum * inv_rms * weight);
    }
}

} // namespace

Status RunAddRmsNormF32Reference(const AddRmsNormF32KernelArgs& args) noexcept {
    for (int64_t row = 0; row < args.row_count; ++row) {
        RunAddRmsNormF32RowReference(args, row);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
