#include "linear_internal.h"

namespace aethermind::cpu::detail {

Status RunLinearF32Reference(const LinearF32KernelArgs& args) noexcept {
    if (args.row_count == 0 || args.out_features == 0) {
        return Status::Ok();
    }

    if (args.in_features == 0) {
        for (int64_t row = 0; row < args.row_count; ++row) {
            float* const output_row = args.output + row * args.output_row_stride;
            for (int64_t out_feature = 0; out_feature < args.out_features; ++out_feature) {
                output_row[out_feature * args.output_col_stride] = 0.0F;
            }
        }
        return Status::Ok();
    }

    for (int64_t row = 0; row < args.row_count; ++row) {
        const float* const input_row = args.input + row * args.input_row_stride;
        float* const output_row = args.output + row * args.output_row_stride;
        for (int64_t out_feature = 0; out_feature < args.out_features; ++out_feature) {
            const float* const weight_row = args.weight + out_feature * args.weight_row_stride;
            double sum = 0.0;
            for (int64_t in_feature = 0; in_feature < args.in_features; ++in_feature) {
                sum += static_cast<double>(input_row[in_feature * args.input_col_stride]) *
                       static_cast<double>(weight_row[in_feature * args.weight_col_stride]);
            }
            output_row[out_feature * args.output_col_stride] = static_cast<float>(sum);
        }
    }

    return Status::Ok();
}

} // namespace aethermind::cpu::detail
