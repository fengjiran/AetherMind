#include "aethermind/base/macros.h"
#include "aethermind/operators/rope_frequency_resolver.h"
#include "rope_internal.h"
#include "utils/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace aethermind::cpu::detail {
namespace {

struct RoPERuntimeState {
    double dynamic_base{};
    uint8_t static_frequency_table{};
};

StatusOr<int64_t> ValidatePosIdsAndGetMaxPos(const RoPEF32KernelArgs& args) noexcept {
    int64_t max_pos = 0;
    for (int64_t token = 0; token < args.seq_len; ++token) {
        const int64_t pos = args.pos_ids[token * args.pos_stride];
        if (pos < 0) {
            return Status::InvalidArgument(
                    "CPU RoPE requires non-negative position_ids");
        }
        max_pos = std::max(max_pos, pos);
    }
    return max_pos;
}

StatusOr<double> DynamicMaxInverseFrequency(const RoPEF32KernelArgs& args,
                                            double dynamic_base) noexcept {
    AM_DCHECK(std::isfinite(dynamic_base));
    AM_DCHECK(dynamic_base > 0.0);

    if (dynamic_base >= 1.0) {
        return 1.0;
    }

    const double last_exponent = -2.0 * static_cast<double>(args.freq_count - 1) /
                                 static_cast<double>(args.rotary_dim);
    const double max_frequency = std::pow(dynamic_base, last_exponent);
    if (!std::isfinite(max_frequency) || max_frequency <= 0.0) {
        return Status::Overflow("CPU RoPE Dynamic NTK inverse frequency is not finite");
    }
    return max_frequency;
}

double FrequencyForPairUnchecked(const RoPEF32KernelArgs& args,
                                 std::span<const std::byte> attrs,
                                 const RoPERuntimeState& runtime,
                                 int64_t pair) noexcept {
    if (args.algorithm != RoPEAlgorithm::kDynamicNtk) {
        return ReadStaticInvFreqUnchecked(args.freq_count, attrs,
                                          runtime.static_frequency_table, pair);
    }

    const double exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(args.rotary_dim);
    return std::pow(runtime.dynamic_base, exponent);
}

double EffectivePositionUnchecked(const RoPEF32KernelArgs& args,
                                  int64_t position) noexcept {
    return static_cast<double>(position) / args.position_divisor;
}

StatusOr<RoPERuntimeState> PrepareRuntimeState(const RoPEF32KernelArgs& args) noexcept {
    AM_ASSIGN_OR_RETURN(const int64_t max_pos, ValidatePosIdsAndGetMaxPos(args));
    if (max_pos == std::numeric_limits<int64_t>::max()) {
        return Status::Overflow("CPU RoPE max position cannot form sequence length");
    }

    const int64_t effective_seq_len = max_pos + 1;
    double dynamic_base = args.theta;
    if (args.algorithm == RoPEAlgorithm::kDynamicNtk) {
        AM_ASSIGN_OR_RETURN(dynamic_base,
                            ComputeDynamicNtkBase(args.theta, args.rotary_dim, args.factor,
                                                  args.original_context_length,
                                                  effective_seq_len));
    }

    const uint8_t static_frequency_table =
            args.algorithm == RoPEAlgorithm::kLongRope &&
                            effective_seq_len > args.original_context_length
                    ? 1
                    : 0;
    double max_inv_freq = static_frequency_table == 0
                                  ? args.short_max_inv_freq
                                  : args.long_max_inv_freq;
    if (args.algorithm == RoPEAlgorithm::kDynamicNtk) {
        AM_ASSIGN_OR_RETURN(max_inv_freq, DynamicMaxInverseFrequency(args, dynamic_base));
    }

    if (const double max_effective_pos = EffectivePositionUnchecked(args, max_pos);
        !std::isfinite(max_effective_pos) || !std::isfinite(max_effective_pos * max_inv_freq)) {
        return Status::Overflow("CPU RoPE angle is not finite");
    }

    return RoPERuntimeState{
            .dynamic_base = dynamic_base,
            .static_frequency_table = static_frequency_table,
    };
}

void RotateHeads(const float* input, float* output,
                 int64_t num_heads, int64_t head_dim, int64_t rotary_dim,
                 RoPEPairing pairing,
                 int64_t input_col_stride, int64_t output_col_stride,
                 int64_t pair,
                 double cosine, double sine) noexcept {
    const int64_t first_pair_offset = pairing == RoPEPairing::kSplitHalf ? pair : pair * 2;
    const int64_t second_pair_offset = pairing == RoPEPairing::kSplitHalf ? rotary_dim / 2 + pair : pair * 2 + 1;
    for (int64_t head = 0; head < num_heads; ++head) {
        const int64_t head_offset = head * head_dim;
        const size_t input_idx0 = (head_offset + first_pair_offset) * input_col_stride;
        const size_t input_idx1 = (head_offset + second_pair_offset) * input_col_stride;
        const size_t output_idx0 = (head_offset + first_pair_offset) * output_col_stride;
        const size_t output_idx1 = (head_offset + second_pair_offset) * output_col_stride;

        const float first = input[input_idx0];
        const float second = input[input_idx1];
        output[output_idx0] = static_cast<float>(
                static_cast<double>(first) * cosine - static_cast<double>(second) * sine);
        output[output_idx1] = static_cast<float>(
                static_cast<double>(second) * cosine + static_cast<double>(first) * sine);
    }
}

void CopyUnrotatedTail(const float* input, float* output,
                       int64_t num_heads, int64_t head_dim, int64_t rotary_dim,
                       int64_t input_col_stride, int64_t output_col_stride) noexcept {
    for (int64_t head = 0; head < num_heads; ++head) {
        const int64_t head_offset = head * head_dim;
        for (int64_t i = rotary_dim; i < head_dim; ++i) {
            output[(head_offset + i) * output_col_stride] = input[(head_offset + i) * input_col_stride];
        }
    }
}

} // namespace

Status RunRoPEF32Reference(const RoPEF32KernelArgs& args,
                           std::span<const std::byte> attrs) noexcept {
    AM_ASSIGN_OR_RETURN(const RoPERuntimeState runtime, PrepareRuntimeState(args));

    for (int64_t pair = 0; pair < args.rotary_dim / 2; ++pair) {
        const double inv_freq = FrequencyForPairUnchecked(args, attrs, runtime, pair);
        for (int64_t token = 0; token < args.seq_len; ++token) {
            const double pos = EffectivePositionUnchecked(args, args.pos_ids[token * args.pos_stride]);
            const double angle = pos * inv_freq;
            const double cosine = std::cos(angle) * args.rotary_output_scale;
            const double sine = std::sin(angle) * args.rotary_output_scale;
            RotateHeads(args.q + token * args.q_row_stride,
                        args.q_output + token * args.q_output_row_stride,
                        args.num_q_heads, args.head_dim, args.rotary_dim, args.pairing,
                        args.q_col_stride, args.q_output_col_stride, pair,
                        cosine, sine);
            RotateHeads(args.k + token * args.k_row_stride,
                        args.k_output + token * args.k_output_row_stride,
                        args.num_kv_heads, args.head_dim, args.rotary_dim, args.pairing,
                        args.k_col_stride, args.k_output_col_stride, pair,
                        cosine, sine);
        }
    }

    for (int64_t token = 0; token < args.seq_len; ++token) {
        CopyUnrotatedTail(args.q + token * args.q_row_stride,
                          args.q_output + token * args.q_output_row_stride,
                          args.num_q_heads, args.head_dim, args.rotary_dim,
                          args.q_col_stride, args.q_output_col_stride);
        CopyUnrotatedTail(args.k + token * args.k_row_stride,
                          args.k_output + token * args.k_output_row_stride,
                          args.num_kv_heads, args.head_dim, args.rotary_dim,
                          args.k_col_stride, args.k_output_col_stride);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
