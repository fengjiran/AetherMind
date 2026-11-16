#include "aethermind/backend/cpu/kernels/common/a16_utils.h"
#include "rmsnorm_internal.h"

#include <cmath>

// Compiled with -ffp-contract=off (see src/CMakeLists.txt): the variance is an
// independent rounded F32 product followed by a rounded F32 add per element.

namespace aethermind::cpu::detail {

Status RunRmsNormA16Reference(const RmsNormA16W32KernelArgs& args) noexcept {
    const size_t act_bytes = A16ElementBytes(args.act_dtype);
    for (int64_t row = 0; row < args.row_count; ++row) {
        const std::byte* const input_row =
                args.input + static_cast<size_t>(row * args.input_row_stride) * act_bytes;
        std::byte* const output_row =
                args.output + static_cast<size_t>(row * args.output_row_stride) * act_bytes;

        float sum_sq = 0.0F;
        for (int64_t j = 0; j < args.hidden_size; ++j) {
            const float x = LoadA16Element(
                    input_row + static_cast<size_t>(j * args.input_col_stride) * act_bytes,
                    args.act_dtype);
            sum_sq = sum_sq + x * x;
        }

        const float mean_sq = sum_sq / static_cast<float>(args.hidden_size);
        const float inv_rms = 1.0F / std::sqrt(mean_sq + args.eps);
        for (int64_t j = 0; j < args.hidden_size; ++j) {
            const float x = LoadA16Element(
                    input_row + static_cast<size_t>(j * args.input_col_stride) * act_bytes,
                    args.act_dtype);
            const float w = args.weight[static_cast<size_t>(j * args.weight_stride)];
            StoreA16Element(
                    output_row + static_cast<size_t>(j * args.output_col_stride) * act_bytes,
                    x * inv_rms * w,
                    args.act_dtype);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
