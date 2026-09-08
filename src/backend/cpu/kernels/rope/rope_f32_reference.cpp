#include "aethermind/base/macros.h"
#include "rope_internal.h"

#include <cmath>

namespace aethermind::cpu::detail {
namespace {

Status ValidatePositionIdsAndAngleRange(const RoPEF32KernelArgs& args) noexcept {
    int64_t max_pos = 0;
    for (int64_t token = 0; token < args.seq_len; ++token) {
        const int64_t position = args.pos_ids[token * args.pos_stride];
        if (position < 0) {
            return Status::InvalidArgument("CPU RoPE requires non-negative position_ids");
        }

        if (position > max_pos) {
            max_pos = position;
        }
    }

    const double max_effective_pos = static_cast<double>(max_pos) / args.pos_divisor;
    if (!std::isfinite(max_effective_pos)) {
        return Status::Overflow("CPU RoPE effective position is not finite");
    }

    if (const double max_angle = max_effective_pos * args.max_inverse_frequency;
        !std::isfinite(max_angle)) {
        return Status::Overflow("CPU RoPE angle is not finite");
    }
    return Status::Ok();
}

void RotateHeads(const float* input,
                 float* output,
                 int64_t num_heads,
                 int64_t head_dim,
                 int64_t input_col_stride,
                 int64_t output_col_stride,
                 int64_t pair,
                 double cosine,
                 double sine) noexcept {
    const int64_t half = head_dim / 2;
    for (int64_t head = 0; head < num_heads; ++head) {
        const int64_t head_offset = head * head_dim;
        const float first = input[(head_offset + pair) * input_col_stride];
        const float second = input[(head_offset + half + pair) * input_col_stride];
        output[(head_offset + pair) * output_col_stride] = static_cast<float>(
                static_cast<double>(first) * cosine - static_cast<double>(second) * sine);
        output[(head_offset + half + pair) * output_col_stride] = static_cast<float>(
                static_cast<double>(second) * cosine + static_cast<double>(first) * sine);
    }
}

} // namespace

Status RunRoPEF32Reference(const RoPEF32KernelArgs& args) noexcept {
    // This pass intentionally precedes all stores. position_ids is mutable
    // runtime data even when the prepared tensor bindings are reused.
    AM_RETURN_IF_ERROR(ValidatePositionIdsAndAngleRange(args));

    const int64_t half = args.head_dim / 2;
    for (int64_t pair = 0; pair < half; ++pair) {
        const double inv_frequency =
                ComputeRoPEInvFreqs(args.theta, args.head_dim, pair);
        for (int64_t token = 0; token < args.seq_len; ++token) {
            const double effective_position =
                    static_cast<double>(args.pos_ids[token * args.pos_stride]) /
                    args.pos_divisor;
            const double angle = effective_position * inv_frequency;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);

            RotateHeads(args.q + token * args.q_row_stride,
                        args.q_output + token * args.q_output_row_stride,
                        args.num_q_heads,
                        args.head_dim,
                        args.q_col_stride,
                        args.q_output_col_stride,
                        pair,
                        cosine,
                        sine);
            RotateHeads(args.k + token * args.k_row_stride,
                        args.k_output + token * args.k_output_row_stride,
                        args.num_kv_heads,
                        args.head_dim,
                        args.k_col_stride,
                        args.k_output_col_stride,
                        pair,
                        cosine,
                        sine);
        }
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
