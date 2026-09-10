#include "aethermind/base/macros.h"
#include "aethermind/operators/rope_frequency_resolver.h"
#include "rope_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace aethermind::cpu::detail {
namespace {

uint8_t ExpectedFrequencyTableCount(RoPEAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case RoPEAlgorithm::kDynamicNtk:
            return 0;
        case RoPEAlgorithm::kLongRope:
            return 2;
        case RoPEAlgorithm::kStandard:
        case RoPEAlgorithm::kLinear:
        case RoPEAlgorithm::kYarn:
        case RoPEAlgorithm::kLlama3:
            return 1;
    }
    return 255;
}

StatusOr<RoPEF32KernelMetadata> ReadAndValidateMetadata(
        std::span<const std::byte> attrs) noexcept {
    if (attrs.size() < sizeof(RoPEF32KernelMetadata)) {
        return Status::InvalidArgument("CPU RoPE metadata attrs are truncated");
    }

    RoPEF32KernelMetadata metadata{};
    std::memcpy(&metadata, attrs.data(), sizeof(metadata));
    if (metadata.rotary_dim <= 0 || metadata.rotary_dim % 2 != 0 ||
        metadata.freq_count != static_cast<uint32_t>(metadata.rotary_dim / 2) ||
        metadata.freq_table_count != ExpectedFrequencyTableCount(metadata.algorithm) ||
        !std::isfinite(metadata.rotary_output_scale)) {
        return Status::InvalidArgument("CPU RoPE metadata attrs are invalid");
    }

    if (metadata.freq_count > 0 && static_cast<size_t>(metadata.freq_count) >
                                           std::numeric_limits<size_t>::max() / sizeof(double)) {
        return Status::InvalidArgument("CPU RoPE frequency attrs overflow");
    }

    size_t table_bytes = static_cast<size_t>(metadata.freq_count) * sizeof(double);
    if (metadata.freq_table_count > 0 && table_bytes > std::numeric_limits<size_t>::max() /
                                                               metadata.freq_table_count) {
        return Status::InvalidArgument("CPU RoPE frequency attrs overflow");
    }

    table_bytes *= metadata.freq_table_count;
    if (table_bytes > attrs.size() - sizeof(metadata) ||
        attrs.size() != sizeof(metadata) + table_bytes) {
        return Status::InvalidArgument("CPU RoPE frequency attrs are truncated");
    }
    return metadata;
}

StatusOr<int64_t> ValidatePositionIdsAndGetEffectiveSequenceLength(
        const RoPEF32KernelArgs& args) noexcept {
    int64_t max_pos = 0;
    for (int64_t token = 0; token < args.seq_len; ++token) {
        const int64_t position = args.pos_ids[token * args.pos_stride];
        if (position < 0) {
            return Status::InvalidArgument("CPU RoPE requires non-negative position_ids");
        }
        max_pos = std::max(max_pos, position);
    }

    if (max_pos == std::numeric_limits<int64_t>::max()) {
        return Status::Overflow("CPU RoPE max position cannot form sequence length");
    }
    return max_pos + 1;
}

double ReadFrequency(const RoPEF32KernelMetadata& metadata,
                     std::span<const std::byte> attrs,
                     uint8_t table,
                     int64_t pair) noexcept {
    double frequency = 0.0;
    const size_t index = static_cast<size_t>(table) * metadata.freq_count +
                         static_cast<size_t>(pair);
    std::memcpy(&frequency, attrs.data() + sizeof(metadata) + index * sizeof(double),
                sizeof(frequency));
    return frequency;
}

StatusOr<double> ResolveDynamicBase(const RoPEF32KernelMetadata& metadata,
                                    int64_t effective_seq_len) noexcept {
    if (metadata.algorithm != RoPEAlgorithm::kDynamicNtk) {
        return metadata.theta;
    }
    return ComputeDynamicNtkBase(metadata.theta, metadata.rotary_dim, metadata.factor,
                                 metadata.original_context_length,
                                 effective_seq_len);
}

StatusOr<double> FrequencyForPair(const RoPEF32KernelMetadata& metadata,
                                  std::span<const std::byte> attrs,
                                  int64_t effective_seq_len,
                                  double dynamic_base,
                                  int64_t pair) noexcept {
    if (metadata.algorithm == RoPEAlgorithm::kDynamicNtk) {
        const double exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(metadata.rotary_dim);
        const double frequency = std::pow(dynamic_base, exponent);
        if (!std::isfinite(frequency) || frequency <= 0.0) {
            return Status::Overflow(
                    "CPU RoPE Dynamic NTK inverse frequency is not finite");
        }
        return frequency;
    }

    const uint8_t table = metadata.algorithm == RoPEAlgorithm::kLongRope &&
                                          effective_seq_len > metadata.original_context_length
                                  ? 1
                                  : 0;
    const double frequency = ReadFrequency(metadata, attrs, table, pair);
    if (!std::isfinite(frequency) || frequency <= 0.0) {
        return Status::Overflow("CPU RoPE inverse frequency is not finite");
    }
    return frequency;
}

StatusOr<double> EffectivePosition(const RoPEF32KernelMetadata& metadata,
                                   int64_t position) noexcept {
    const double result = metadata.algorithm == RoPEAlgorithm::kLinear
                                  ? static_cast<double>(position) / metadata.factor
                                  : static_cast<double>(position);
    if (!std::isfinite(result)) {
        return Status::Overflow("CPU RoPE effective position is not finite");
    }
    return result;
}

Status ValidateAngleRange(const RoPEF32KernelArgs& args,
                          const RoPEF32KernelMetadata& metadata,
                          std::span<const std::byte> attrs,
                          int64_t effective_seq_len,
                          double dynamic_base) noexcept {
    int64_t max_pos = 0;
    for (int64_t token = 0; token < args.seq_len; ++token) {
        max_pos = std::max(max_pos, args.pos_ids[token * args.pos_stride]);
    }

    double max_freq = 0.0;
    for (int64_t pair = 0; pair < args.rotary_dim / 2; ++pair) {
        AM_ASSIGN_OR_RETURN(const double frequency,
                            FrequencyForPair(metadata, attrs, effective_seq_len,
                                             dynamic_base, pair));
        max_freq = std::max(max_freq, frequency);
    }

    AM_ASSIGN_OR_RETURN(const double max_effective_position, EffectivePosition(metadata, max_pos));
    if (!std::isfinite(max_effective_position * max_freq)) {
        return Status::Overflow("CPU RoPE angle is not finite");
    }
    return Status::Ok();
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

void CopyUnrotatedTail(const float* input,
                       float* output,
                       int64_t num_heads,
                       int64_t head_dim,
                       int64_t rotary_dim,
                       int64_t input_col_stride,
                       int64_t output_col_stride) noexcept {
    for (int64_t head = 0; head < num_heads; ++head) {
        const int64_t head_offset = head * head_dim;
        for (int64_t element = rotary_dim; element < head_dim; ++element) {
            output[(head_offset + element) * output_col_stride] =
                    input[(head_offset + element) * input_col_stride];
        }
    }
}

} // namespace

Status RunRoPEF32Reference(const RoPEF32KernelArgs& args,
                           std::span<const std::byte> attrs) noexcept {
    AM_ASSIGN_OR_RETURN(const RoPEF32KernelMetadata metadata, ReadAndValidateMetadata(attrs));
    if (metadata.head_dim != args.head_dim || metadata.rotary_dim != args.rotary_dim ||
        metadata.pairing != args.pairing) {
        return Status::InvalidArgument("CPU RoPE prepared args do not match metadata");
    }

    AM_ASSIGN_OR_RETURN(const int64_t effective_seq_len,
                        ValidatePositionIdsAndGetEffectiveSequenceLength(args));
    AM_ASSIGN_OR_RETURN(const double dynamic_base,
                        ResolveDynamicBase(metadata, effective_seq_len));
    AM_RETURN_IF_ERROR(ValidateAngleRange(args, metadata, attrs, effective_seq_len, dynamic_base));

    for (int64_t pair = 0; pair < args.rotary_dim / 2; ++pair) {
        AM_ASSIGN_OR_RETURN(const double inv_freq,
                            FrequencyForPair(metadata, attrs, effective_seq_len,
                                             dynamic_base, pair));
        for (int64_t token = 0; token < args.seq_len; ++token) {
            AM_ASSIGN_OR_RETURN(const double pos,
                                EffectivePosition(metadata,
                                                  args.pos_ids[token * args.pos_stride]));
            const double angle = pos * inv_freq;
            const double cosine = std::cos(angle) * metadata.rotary_output_scale;
            const double sine = std::sin(angle) * metadata.rotary_output_scale;
            RotateHeads(args.q + token * args.q_row_stride,
                        args.q_output + token * args.q_output_row_stride,
                        args.num_q_heads, args.head_dim, args.rotary_dim, args.pairing,
                        args.q_col_stride, args.q_output_col_stride, pair, cosine, sine);
            RotateHeads(args.k + token * args.k_row_stride,
                        args.k_output + token * args.k_output_row_stride,
                        args.num_kv_heads, args.head_dim, args.rotary_dim, args.pairing,
                        args.k_col_stride, args.k_output_col_stride, pair, cosine, sine);
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
