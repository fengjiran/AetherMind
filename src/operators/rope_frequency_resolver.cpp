#include "aethermind/operators/rope_frequency_resolver.h"
#include "aethermind/base/macros.h"
#include "utils/variant_utils.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <string_view>

namespace aethermind {
namespace {

bool IsFinitePositive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

Status InvalidAlgorithmParameter(std::string_view name) {
    return Status::InvalidArgument("RoPE " + std::string(name) +
                                   " must be finite and positive");
}

Status ValidateCommon(const RoPEParams& params) {
    if (const int64_t rotary_dim = EffectiveRoPERotaryDim(params);
        params.head_dim <= 0 || rotary_dim <= 0 || rotary_dim > params.head_dim ||
        rotary_dim % 2 != 0 || !IsFinitePositive(params.theta)) {
        return Status::InvalidArgument("RoPE frequency resolver received invalid geometry");
    }
    return Status::Ok();
}

StatusOr<std::vector<double>> MakeBaseFreqs(double base, int64_t rotary_dim) {
    const int64_t pair_count = rotary_dim / 2;
    std::vector<double> result;
    result.reserve(static_cast<size_t>(pair_count));
    for (int64_t i = 0; i < pair_count; ++i) {
        const double exponent = -2.0 * static_cast<double>(i) / static_cast<double>(rotary_dim);
        const double value = std::pow(base, exponent);
        if (!IsFinitePositive(value)) {
            return Status::Overflow("RoPE inverse frequency is not finite");
        }
        result.push_back(value);
    }
    return result;
}

double YarnCorrectionDim(double rotations, int64_t rotary_dim, double theta,
                         int64_t original_context_length) noexcept {
    return static_cast<double>(rotary_dim) *
           std::log(static_cast<double>(original_context_length) / (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(theta));
}

StatusOr<ResolvedRoPEFreqs> ResolveYarn(const RoPEParams& params, const YarnRoPE& algorithm) {
    const int64_t rotary_dim = EffectiveRoPERotaryDim(params);
    AM_ASSIGN_OR_RETURN(auto base, MakeBaseFreqs(params.theta, rotary_dim));
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
    for (size_t i = 0; i < base.size(); ++i) {
        const double ramp = std::clamp((static_cast<double>(i) - low) / denominator, 0.0, 1.0);
        base[i] = (1.0 - ramp) * base[i] + ramp * (base[i] / algorithm.factor);
        if (!IsFinitePositive(base[i])) {
            return Status::Overflow("RoPE YaRN inverse frequency is not finite");
        }
    }

    return ResolvedRoPEFreqs{
            .inv_freqs = std::move(base),
            .rotary_output_scale = algorithm.rotary_output_scale,
    };
}

StatusOr<ResolvedRoPEFreqs> ResolveLlama3(const RoPEParams& params,
                                          const Llama3RoPE& algorithm) {
    AM_ASSIGN_OR_RETURN(auto base, MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
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
    return ResolvedRoPEFreqs{.inv_freqs = std::move(base)};
}

StatusOr<ResolvedRoPEFreqs> ResolveDynamicNtk(const RoPEParams& params,
                                              const DynamicNtkRoPE& algorithm,
                                              int64_t effective_sequence_length) {
    AM_ASSIGN_OR_RETURN(const double base, ComputeDynamicNtkBase(
                                                   params.theta, EffectiveRoPERotaryDim(params), algorithm.factor,
                                                   algorithm.original_context_length, effective_sequence_length));
    AM_ASSIGN_OR_RETURN(auto frequencies,
                        MakeBaseFreqs(base, EffectiveRoPERotaryDim(params)));
    return ResolvedRoPEFreqs{.inv_freqs = std::move(frequencies)};
}

StatusOr<ResolvedRoPEFreqs> ResolveLongRope(const RoPEParams& params,
                                            const LongRoPE& algorithm,
                                            int64_t effective_sequence_length) {
    const bool use_long = effective_sequence_length > algorithm.original_context_length;
    const std::vector<double>& factors = use_long ? algorithm.long_factors : algorithm.short_factors;
    AM_ASSIGN_OR_RETURN(auto frequencies, MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
    for (size_t pair = 0; pair < frequencies.size(); ++pair) {
        frequencies[pair] /= factors[pair];
        if (!IsFinitePositive(frequencies[pair])) {
            return Status::Overflow("RoPE LongRoPE inverse frequency is not finite");
        }
    }
    return ResolvedRoPEFreqs{
            .inv_freqs = std::move(frequencies),
            .rotary_output_scale = algorithm.rotary_output_scale,
    };
}

} // namespace

Status ValidateRoPEFrequencyParameters(const RoPEParams& params) {
    AM_RETURN_IF_ERROR(ValidateCommon(params));
    const int64_t pair_count = EffectiveRoPERotaryDim(params) / 2;
    auto visitor = overloaded{
            [](const StandardRoPE&) -> Status { return Status::Ok(); },
            [](const LinearRoPE& algorithm) -> Status {
                return IsFinitePositive(algorithm.factor)
                               ? Status::Ok()
                               : InvalidAlgorithmParameter("Linear factor");
            },
            [&](const DynamicNtkRoPE& algorithm) -> Status {
                if (!IsFinitePositive(algorithm.factor)) {
                    return InvalidAlgorithmParameter("Dynamic NTK factor");
                }
                if (algorithm.original_context_length <= 0 ||
                    EffectiveRoPERotaryDim(params) <= 2) {
                    return Status::InvalidArgument(
                            "RoPE Dynamic NTK requires positive original context and rotary_dim > 2");
                }
                return Status::Ok();
            },
            [&](const YarnRoPE& algorithm) -> Status {
                if (!IsFinitePositive(algorithm.factor) ||
                    params.theta <= 1.0 ||
                    !IsFinitePositive(algorithm.beta_fast) ||
                    !IsFinitePositive(algorithm.beta_slow) ||
                    !IsFinitePositive(algorithm.rotary_output_scale) ||
                    algorithm.original_context_length <= 0 ||
                    algorithm.beta_fast <= algorithm.beta_slow) {
                    return Status::InvalidArgument("RoPE YaRN parameters are invalid");
                }
                return Status::Ok();
            },
            [](const Llama3RoPE& algorithm) -> Status {
                if (!IsFinitePositive(algorithm.factor) ||
                    !IsFinitePositive(algorithm.low_frequency_factor) ||
                    !IsFinitePositive(algorithm.high_frequency_factor) ||
                    algorithm.high_frequency_factor <= algorithm.low_frequency_factor ||
                    algorithm.original_context_length <= 0) {
                    return Status::InvalidArgument("RoPE Llama3 parameters are invalid");
                }
                return Status::Ok();
            },
            [&](const LongRoPE& algorithm) -> Status {
                if (algorithm.original_context_length <= 0 ||
                    !IsFinitePositive(algorithm.rotary_output_scale) ||
                    algorithm.short_factors.size() != static_cast<size_t>(pair_count) ||
                    algorithm.long_factors.size() != static_cast<size_t>(pair_count)) {
                    return Status::InvalidArgument("RoPE LongRoPE parameters are invalid");
                }

                for (double value: algorithm.short_factors) {
                    if (!IsFinitePositive(value)) {
                        return InvalidAlgorithmParameter("LongRoPE short factor");
                    }
                }
                for (double value: algorithm.long_factors) {
                    if (!IsFinitePositive(value)) {
                        return InvalidAlgorithmParameter("LongRoPE long factor");
                    }
                }
                return Status::Ok();
            }};
    return std::visit(visitor, params.algorithm);
}

bool IsDynamicRoPE(const RoPEAlgorithmParams& params) noexcept {
    return std::holds_alternative<DynamicNtkRoPE>(params) ||
           std::holds_alternative<LongRoPE>(params);
}

StatusOr<double> ComputeDynamicNtkBase(double theta,
                                       int64_t rotary_dim,
                                       double factor,
                                       int64_t original_context_len,
                                       int64_t effective_seq_len) {
    if (!IsFinitePositive(theta) || !IsFinitePositive(factor) || rotary_dim <= 2 ||
        original_context_len <= 0 || effective_seq_len <= 0) {
        return Status::InvalidArgument("RoPE Dynamic NTK parameters are invalid");
    }

    const auto seq_len = static_cast<double>(std::max(effective_seq_len, original_context_len));
    const double ratio = factor * seq_len / static_cast<double>(original_context_len) - (factor - 1.0);
    const double exponent = static_cast<double>(rotary_dim) / static_cast<double>(rotary_dim - 2);
    const double base = theta * std::pow(ratio, exponent);
    if (!IsFinitePositive(base)) {
        return Status::Overflow("RoPE Dynamic NTK base frequency is not finite");
    }
    return base;
}

StatusOr<ResolvedRoPEFreqs> ResolveStaticRoPEFrequencies(const RoPEParams& params) {
    AM_RETURN_IF_ERROR(ValidateRoPEFrequencyParameters(params));
    auto visitor = overloaded{
            [&](const StandardRoPE&) -> StatusOr<ResolvedRoPEFreqs> {
                AM_ASSIGN_OR_RETURN(auto frequencies,
                                    MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
                return ResolvedRoPEFreqs{.inv_freqs = std::move(frequencies)};
            },
            [&](const LinearRoPE& algorithm) -> StatusOr<ResolvedRoPEFreqs> {
                AM_ASSIGN_OR_RETURN(auto frequencies,
                                    MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
                for (double& value: frequencies) value /= algorithm.factor;
                return ResolvedRoPEFreqs{.inv_freqs = std::move(frequencies)};
            },
            [&](const YarnRoPE& algorithm) -> StatusOr<ResolvedRoPEFreqs> {
                return ResolveYarn(params, algorithm);
            },
            [&](const Llama3RoPE& algorithm) -> StatusOr<ResolvedRoPEFreqs> {
                return ResolveLlama3(params, algorithm);
            },
            [](const DynamicNtkRoPE&) -> StatusOr<ResolvedRoPEFreqs> {
                return Status::InvalidArgument(
                        "RoPE algorithm requires execution-time sequence length");
            },
            [](const LongRoPE&) -> StatusOr<ResolvedRoPEFreqs> {
                return Status::InvalidArgument(
                        "RoPE algorithm requires execution-time sequence length");
            }};
    return std::visit(visitor, params.algorithm);
}

StatusOr<ResolvedRoPEFreqs> ResolveDynamicRoPEFrequencies(const RoPEParams& params,
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
