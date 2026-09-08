#include "aethermind/operators/rope_frequency_resolver.h"

#include "aethermind/base/macros.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <string_view>
#include <type_traits>

namespace aethermind {
namespace {

bool IsFinitePositive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

Status InvalidAlgorithmParameter(std::string_view name) {
    return Status::InvalidArgument("RoPE " + std::string(name) + " must be finite and positive");
}

Status ValidateCommon(const RoPEParams& params) {
    const int64_t rotary_dim = EffectiveRoPERotaryDim(params);
    if (params.head_dim <= 0 || rotary_dim <= 0 || rotary_dim > params.head_dim ||
        rotary_dim % 2 != 0 || !IsFinitePositive(params.theta)) {
        return Status::InvalidArgument("RoPE frequency resolver received invalid geometry");
    }
    return Status::Ok();
}

StatusOr<std::vector<double>> MakeBaseFrequencies(const RoPEParams& params,
                                                  double base) {
    const int64_t rotary_dim = EffectiveRoPERotaryDim(params);
    const int64_t pair_count = rotary_dim / 2;
    std::vector<double> result;
    result.reserve(static_cast<size_t>(pair_count));
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        const double exponent = -2.0 * static_cast<double>(pair) /
                                static_cast<double>(rotary_dim);
        const double value = std::pow(base, exponent);
        if (!IsFinitePositive(value)) {
            return Status::Overflow("RoPE inverse frequency is not finite");
        }
        result.push_back(value);
    }
    return result;
}

double YarnCorrectionDim(double rotations,
                         int64_t rotary_dim,
                         double theta,
                         int64_t original_context_length) noexcept {
    return static_cast<double>(rotary_dim) *
           std::log(static_cast<double>(original_context_length) /
                    (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(theta));
}

StatusOr<ResolvedRoPEFrequencies> ResolveYarn(const RoPEParams& params,
                                              const YarnRoPE& algorithm) {
    AM_ASSIGN_OR_RETURN(auto base, MakeBaseFrequencies(params, params.theta));
    const int64_t rotary_dim = EffectiveRoPERotaryDim(params);
    double low = YarnCorrectionDim(algorithm.beta_fast, rotary_dim, params.theta,
                                   algorithm.original_context_length);
    double high = YarnCorrectionDim(algorithm.beta_slow, rotary_dim, params.theta,
                                    algorithm.original_context_length);
    if (algorithm.truncate_correction_range) {
        low = std::floor(low);
        high = std::ceil(high);
    }
    // Match HF's correction-range clamp. `pair` spans [0, rotary_dim / 2),
    // while the historical reference clamps the range against rotary_dim - 1.
    low = std::clamp(low, 0.0, static_cast<double>(rotary_dim - 1));
    high = std::clamp(high, 0.0, static_cast<double>(rotary_dim - 1));
    const double denominator = std::max(high - low, 1.0e-12);
    for (size_t pair = 0; pair < base.size(); ++pair) {
        const double ramp = std::clamp((static_cast<double>(pair) - low) / denominator,
                                       0.0, 1.0);
        base[pair] = (1.0 - ramp) * base[pair] + ramp * (base[pair] / algorithm.factor);
        if (!IsFinitePositive(base[pair])) {
            return Status::Overflow("RoPE YaRN inverse frequency is not finite");
        }
    }
    return ResolvedRoPEFrequencies{
            .inverse_frequencies = std::move(base),
            .attention_scale = algorithm.attention_scale,
    };
}

StatusOr<ResolvedRoPEFrequencies> ResolveLlama3(const RoPEParams& params,
                                                const Llama3RoPE& algorithm) {
    AM_ASSIGN_OR_RETURN(auto base, MakeBaseFrequencies(params, params.theta));
    const double low_wavelength = static_cast<double>(algorithm.original_context_length) /
                                  algorithm.low_frequency_factor;
    const double high_wavelength = static_cast<double>(algorithm.original_context_length) /
                                   algorithm.high_frequency_factor;
    for (double& frequency: base) {
        const double wavelength = 2.0 * std::numbers::pi / frequency;
        if (wavelength > low_wavelength) {
            frequency /= algorithm.factor;
        } else if (wavelength >= high_wavelength) {
            const double smooth =
                    (static_cast<double>(algorithm.original_context_length) / wavelength -
                     algorithm.low_frequency_factor) /
                    (algorithm.high_frequency_factor - algorithm.low_frequency_factor);
            frequency = (1.0 - smooth) * frequency / algorithm.factor + smooth * frequency;
        }
        if (!IsFinitePositive(frequency)) {
            return Status::Overflow("RoPE Llama3 inverse frequency is not finite");
        }
    }
    return ResolvedRoPEFrequencies{.inverse_frequencies = std::move(base)};
}

StatusOr<ResolvedRoPEFrequencies> ResolveDynamicNtk(const RoPEParams& params,
                                                    const DynamicNtkRoPE& algorithm,
                                                    int64_t effective_sequence_length) {
    AM_ASSIGN_OR_RETURN(const double base, ComputeDynamicNtkBase(
                                                   params.theta, EffectiveRoPERotaryDim(params), algorithm.factor,
                                                   algorithm.original_context_length, effective_sequence_length));
    AM_ASSIGN_OR_RETURN(auto frequencies, MakeBaseFrequencies(params, base));
    return ResolvedRoPEFrequencies{.inverse_frequencies = std::move(frequencies)};
}

StatusOr<ResolvedRoPEFrequencies> ResolveLongRope(const RoPEParams& params,
                                                  const LongRoPE& algorithm,
                                                  int64_t effective_sequence_length) {
    const bool use_long = effective_sequence_length > algorithm.original_context_length;
    const std::vector<double>& factors = use_long ? algorithm.long_factors : algorithm.short_factors;
    AM_ASSIGN_OR_RETURN(auto frequencies, MakeBaseFrequencies(params, params.theta));
    for (size_t pair = 0; pair < frequencies.size(); ++pair) {
        frequencies[pair] /= factors[pair];
        if (!IsFinitePositive(frequencies[pair])) {
            return Status::Overflow("RoPE LongRoPE inverse frequency is not finite");
        }
    }
    return ResolvedRoPEFrequencies{
            .inverse_frequencies = std::move(frequencies),
            .attention_scale = algorithm.attention_scale,
    };
}

} // namespace

Status ValidateRoPEFrequencyParameters(const RoPEParams& params) {
    AM_RETURN_IF_ERROR(ValidateCommon(params));
    const int64_t pair_count = EffectiveRoPERotaryDim(params) / 2;
    return std::visit(
            [&](const auto& algorithm) -> Status {
                using T = std::decay_t<decltype(algorithm)>;
                if constexpr (std::is_same_v<T, StandardRoPE>) {
                    return Status::Ok();
                } else if constexpr (std::is_same_v<T, LinearRoPE>) {
                    return IsFinitePositive(algorithm.factor)
                                   ? Status::Ok()
                                   : InvalidAlgorithmParameter("Linear factor");
                } else if constexpr (std::is_same_v<T, DynamicNtkRoPE>) {
                    if (!IsFinitePositive(algorithm.factor)) {
                        return InvalidAlgorithmParameter("Dynamic NTK factor");
                    }
                    if (algorithm.original_context_length <= 0 ||
                        EffectiveRoPERotaryDim(params) <= 2) {
                        return Status::InvalidArgument(
                                "RoPE Dynamic NTK requires positive original context and rotary_dim > 2");
                    }
                    return Status::Ok();
                } else if constexpr (std::is_same_v<T, YarnRoPE>) {
                    if (!IsFinitePositive(algorithm.factor) ||
                        params.theta <= 1.0 ||
                        !IsFinitePositive(algorithm.beta_fast) ||
                        !IsFinitePositive(algorithm.beta_slow) ||
                        !IsFinitePositive(algorithm.attention_scale) ||
                        algorithm.original_context_length <= 0 ||
                        algorithm.beta_fast <= algorithm.beta_slow) {
                        return Status::InvalidArgument("RoPE YaRN parameters are invalid");
                    }
                    return Status::Ok();
                } else if constexpr (std::is_same_v<T, Llama3RoPE>) {
                    if (!IsFinitePositive(algorithm.factor) ||
                        !IsFinitePositive(algorithm.low_frequency_factor) ||
                        !IsFinitePositive(algorithm.high_frequency_factor) ||
                        algorithm.high_frequency_factor <= algorithm.low_frequency_factor ||
                        algorithm.original_context_length <= 0) {
                        return Status::InvalidArgument("RoPE Llama3 parameters are invalid");
                    }
                    return Status::Ok();
                } else {
                    if (algorithm.original_context_length <= 0 ||
                        !IsFinitePositive(algorithm.attention_scale) ||
                        algorithm.short_factors.size() != static_cast<size_t>(pair_count) ||
                        algorithm.long_factors.size() != static_cast<size_t>(pair_count)) {
                        return Status::InvalidArgument("RoPE LongRoPE parameters are invalid");
                    }
                    for (double value: algorithm.short_factors) {
                        if (!IsFinitePositive(value)) return InvalidAlgorithmParameter("LongRoPE short factor");
                    }
                    for (double value: algorithm.long_factors) {
                        if (!IsFinitePositive(value)) return InvalidAlgorithmParameter("LongRoPE long factor");
                    }
                    return Status::Ok();
                }
            },
            params.algorithm);
}

bool IsDynamicRoPE(const RoPEAlgorithmParams& params) noexcept {
    return std::holds_alternative<DynamicNtkRoPE>(params) ||
           std::holds_alternative<LongRoPE>(params);
}

StatusOr<double> ComputeDynamicNtkBase(double theta,
                                       int64_t rotary_dim,
                                       double factor,
                                       int64_t original_context_length,
                                       int64_t effective_sequence_length) {
    if (!IsFinitePositive(theta) || !IsFinitePositive(factor) || rotary_dim <= 2 ||
        original_context_length <= 0 || effective_sequence_length <= 0) {
        return Status::InvalidArgument("RoPE Dynamic NTK parameters are invalid");
    }
    const double sequence_length = static_cast<double>(std::max(
            effective_sequence_length, original_context_length));
    const double ratio = factor * sequence_length /
                                 static_cast<double>(original_context_length) -
                         (factor - 1.0);
    const double exponent = static_cast<double>(rotary_dim) /
                            static_cast<double>(rotary_dim - 2);
    const double base = theta * std::pow(ratio, exponent);
    if (!IsFinitePositive(base)) {
        return Status::Overflow("RoPE Dynamic NTK base frequency is not finite");
    }
    return base;
}

StatusOr<ResolvedRoPEFrequencies> ResolveStaticRoPEFrequencies(const RoPEParams& params) {
    AM_RETURN_IF_ERROR(ValidateRoPEFrequencyParameters(params));
    return std::visit(
            [&](const auto& algorithm) -> StatusOr<ResolvedRoPEFrequencies> {
                using T = std::decay_t<decltype(algorithm)>;
                if constexpr (std::is_same_v<T, StandardRoPE>) {
                    AM_ASSIGN_OR_RETURN(auto frequencies, MakeBaseFrequencies(params, params.theta));
                    return ResolvedRoPEFrequencies{.inverse_frequencies = std::move(frequencies)};
                } else if constexpr (std::is_same_v<T, LinearRoPE>) {
                    AM_ASSIGN_OR_RETURN(auto frequencies, MakeBaseFrequencies(params, params.theta));
                    for (double& value: frequencies) value /= algorithm.factor;
                    return ResolvedRoPEFrequencies{.inverse_frequencies = std::move(frequencies)};
                } else if constexpr (std::is_same_v<T, YarnRoPE>) {
                    return ResolveYarn(params, algorithm);
                } else if constexpr (std::is_same_v<T, Llama3RoPE>) {
                    return ResolveLlama3(params, algorithm);
                } else {
                    return Status::InvalidArgument("RoPE algorithm requires execution-time sequence length");
                }
            },
            params.algorithm);
}

StatusOr<ResolvedRoPEFrequencies> ResolveDynamicRoPEFrequencies(
        const RoPEParams& params,
        int64_t effective_sequence_length) {
    AM_RETURN_IF_ERROR(ValidateRoPEFrequencyParameters(params));
    if (effective_sequence_length <= 0) {
        return Status::InvalidArgument("RoPE effective sequence length must be positive");
    }
    if (!IsDynamicRoPE(params.algorithm)) {
        return ResolveStaticRoPEFrequencies(params);
    }
    if (const auto* dynamic_ntk = std::get_if<DynamicNtkRoPE>(&params.algorithm)) {
        return ResolveDynamicNtk(params, *dynamic_ntk, effective_sequence_length);
    }
    return ResolveLongRope(params, std::get<LongRoPE>(params.algorithm), effective_sequence_length);
}

} // namespace aethermind
