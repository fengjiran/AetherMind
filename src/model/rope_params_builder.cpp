#include "rope_params_builder.h"
#include "aethermind/operators/rope_frequency_resolver.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind::detail {
namespace {

static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kStandard) == static_cast<uint8_t>(RoPEAlgorithm::kStandard));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kLinear) == static_cast<uint8_t>(RoPEAlgorithm::kLinear));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kDynamicNtk) == static_cast<uint8_t>(RoPEAlgorithm::kDynamicNtk));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kYarn) == static_cast<uint8_t>(RoPEAlgorithm::kYarn));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kLlama3) == static_cast<uint8_t>(RoPEAlgorithm::kLlama3));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kLongRope) == static_cast<uint8_t>(RoPEAlgorithm::kLongRope));

StatusOr<double> RequireRopeDouble(const std::optional<double>& value,
                                   std::string_view field) {
    if (!value.has_value() || !std::isfinite(*value)) {
        return Status::InvalidArgument("MakeRoPEParams: RoPE '" + std::string(field) +
                                       "' must be provided and finite");
    }
    return *value;
}

StatusOr<int64_t> RequireRopeContext(const std::optional<int64_t>& value) {
    if (!value.has_value() || *value <= 0) {
        return Status::InvalidArgument(
                "MakeRoPEParams: RoPE 'original_max_position_embeddings' "
                "must be provided and positive");
    }
    return *value;
}

StatusOr<int64_t> ResolveHfRotaryDim(const HfRopeConfig& rope, int64_t head_dim) {
    if (!rope.partial_rotary_factor.has_value()) {
        return rope.rotary_dim.value_or(head_dim);
    }

    const double factor = *rope.partial_rotary_factor;
    const double dimension = factor * static_cast<double>(head_dim);
    if (!std::isfinite(factor) || factor <= 0.0 || factor > 1.0 ||
        !std::isfinite(dimension) || dimension < 1.0 ||
        dimension > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return Status::InvalidArgument("MakeRoPEParams: invalid partial_rotary_factor");
    }

    const auto derived = static_cast<int64_t>(dimension);
    if (rope.rotary_dim.has_value() && *rope.rotary_dim != derived) {
        return Status::InvalidArgument(
                "MakeRoPEParams: rotary_dim conflicts with partial_rotary_factor");
    }
    return derived;
}

double Mscale(double factor, double multiplier) noexcept {
    return factor <= 1.0 || multiplier == 0.0 ? 1.0 : 0.1 * multiplier * std::log(factor) + 1.0;
}

} // namespace

StatusOr<RoPEParams> MakeRoPEParams(const HfModelConfig& config, int64_t head_dim) {
    AM_ASSIGN_OR_RETURN(const int64_t rotary_dim, ResolveHfRotaryDim(config.rope, head_dim));
    RoPEAlgorithmParams algorithm = StandardRoPE{};
    switch (config.rope.algorithm) {
        case HfRoPEAlgorithm::kStandard:
            break;
        case HfRoPEAlgorithm::kLinear: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            algorithm = LinearRoPE{.factor = factor};
            break;
        }
        case HfRoPEAlgorithm::kDynamicNtk: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            algorithm = DynamicNtkRoPE{
                    .factor = factor,
                    .original_context_length = config.rope.original_context_length.value_or(
                            config.max_position_embeddings)};
            break;
        }
        case HfRoPEAlgorithm::kYarn: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            AM_ASSIGN_OR_RETURN(const int64_t original_context,
                                RequireRopeContext(config.rope.original_context_length));
            double rotary_output_scale = Mscale(factor, 1.0);
            if (config.rope.attention_factor.has_value()) {
                rotary_output_scale = *config.rope.attention_factor;
            } else if (config.rope.mscale.has_value() && config.rope.mscale_all_dim.has_value() &&
                       *config.rope.mscale != 0.0 && *config.rope.mscale_all_dim != 0.0) {
                rotary_output_scale = Mscale(factor, *config.rope.mscale) /
                                      Mscale(factor, *config.rope.mscale_all_dim);
            }
            algorithm = YarnRoPE{.factor = factor,
                                 .original_context_length = original_context,
                                 .beta_fast = config.rope.beta_fast.value_or(32.0),
                                 .beta_slow = config.rope.beta_slow.value_or(1.0),
                                 .rotary_output_scale = rotary_output_scale,
                                 .truncate_correction_range =
                                         config.rope.truncate_correction_range.value_or(true)};
            break;
        }
        case HfRoPEAlgorithm::kLlama3: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            AM_ASSIGN_OR_RETURN(const int64_t original_context,
                                RequireRopeContext(config.rope.original_context_length));
            AM_ASSIGN_OR_RETURN(const double low,
                                RequireRopeDouble(config.rope.low_frequency_factor,
                                                  "low_freq_factor"));
            AM_ASSIGN_OR_RETURN(const double high,
                                RequireRopeDouble(config.rope.high_frequency_factor,
                                                  "high_freq_factor"));
            algorithm = Llama3RoPE{.factor = factor,
                                   .low_frequency_factor = low,
                                   .high_frequency_factor = high,
                                   .original_context_length = original_context};
            break;
        }
        case HfRoPEAlgorithm::kLongRope:
        case HfRoPEAlgorithm::kSu: {
            AM_ASSIGN_OR_RETURN(const int64_t original_context,
                                RequireRopeContext(config.rope.original_context_length));
            if (config.rope.short_factors.empty() || config.rope.long_factors.empty()) {
                return Status::InvalidArgument("MakeRoPEParams: LongRoPE "
                                               "requires short_factor and long_factor arrays");
            }

            double rotary_output_scale = 1.0;
            if (config.rope.attention_factor.has_value()) {
                rotary_output_scale = *config.rope.attention_factor;
            } else if (config.rope.factor.has_value()) {
                const double factor = *config.rope.factor;
                rotary_output_scale = std::sqrt(
                        1.0 + std::log(factor) / std::log(static_cast<double>(original_context)));
            }
            algorithm = LongRoPE{.short_factors = config.rope.short_factors,
                                 .long_factors = config.rope.long_factors,
                                 .original_context_length = original_context,
                                 .rotary_output_scale = rotary_output_scale};
            break;
        }
        case HfRoPEAlgorithm::kUnknown:
            return Status::InvalidArgument(
                    "MakeRoPEParams: HF RoPE algorithm '" +
                    std::string(ToString(config.rope.algorithm)) + "' is not supported");
    }
    RoPEParams result{
            .head_dim = head_dim,
            .rotary_dim = rotary_dim,
            .num_q_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .max_pos_embeddings = config.max_position_embeddings,
            .theta = config.rope.theta,
            .algorithm = std::move(algorithm),
    };
    AM_RETURN_IF_ERROR(ValidateRoPEFreqParams(result));
    return result;
}

} // namespace aethermind::detail