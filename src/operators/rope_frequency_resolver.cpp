#include "aethermind/operators/rope_frequency_resolver.h"
#include "aethermind/base/macros.h"
#include "utils/numeric_utils.h"
#include "utils/variant_utils.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <string_view>

namespace aethermind {
namespace {

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

StatusOr<RoPERotationCoefficients> ResolveYarn(const RoPEParams& params, const YarnRoPE& algorithm) {
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

    return RoPERotationCoefficients{
            .inv_freqs = std::move(base),
            .rotary_output_scale = algorithm.rotary_output_scale,
    };
}

StatusOr<RoPERotationCoefficients> ResolveLlama3(const RoPEParams& params, const Llama3RoPE& algorithm) {
    AM_ASSIGN_OR_RETURN(auto base, MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
    const double low_wavelength = static_cast<double>(algorithm.original_context_length) /
                                  algorithm.low_frequency_factor;
    const double high_wavelength = static_cast<double>(algorithm.original_context_length) /
                                   algorithm.high_frequency_factor;

    for (auto& freq: base) {
        if (const double wavelength = 2.0 * std::numbers::pi / freq;
            wavelength > low_wavelength) {
            freq /= algorithm.factor;
        } else if (wavelength >= high_wavelength) {
            const double smooth =
                    (static_cast<double>(algorithm.original_context_length) / wavelength -
                     algorithm.low_frequency_factor) /
                    (algorithm.high_frequency_factor - algorithm.low_frequency_factor);
            freq = (1.0 - smooth) * freq / algorithm.factor + smooth * freq;
        }

        if (!IsFinitePositive(freq)) {
            return Status::Overflow("RoPE Llama3 inverse frequency is not finite");
        }
    }
    return RoPERotationCoefficients{.inv_freqs = std::move(base)};
}

StatusOr<RoPERotationCoefficients> ResolveDynamicNtk(const RoPEParams& params,
                                                     const DynamicNtkRoPE& algorithm,
                                                     int64_t effective_seq_len) {
    AM_ASSIGN_OR_RETURN(const double base,
                        ComputeDynamicNtkBase(
                                params.theta, EffectiveRoPERotaryDim(params), algorithm.factor,
                                algorithm.original_context_length, effective_seq_len));
    AM_ASSIGN_OR_RETURN(auto frequencies,
                        MakeBaseFreqs(base, EffectiveRoPERotaryDim(params)));
    return RoPERotationCoefficients{.inv_freqs = std::move(frequencies)};
}

StatusOr<RoPERotationCoefficients> ResolveLongRope(const RoPEParams& params,
                                                   const LongRoPE& algorithm,
                                                   int64_t effective_seq_len) {
    const bool use_long = effective_seq_len > algorithm.original_context_length;
    const auto& factors = use_long ? algorithm.long_factors : algorithm.short_factors;
    AM_ASSIGN_OR_RETURN(auto freqs,
                        MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
    for (size_t pair = 0; pair < freqs.size(); ++pair) {
        freqs[pair] /= factors[pair];
        if (!IsFinitePositive(freqs[pair])) {
            return Status::Overflow("RoPE LongRoPE inverse frequency is not finite");
        }
    }

    return RoPERotationCoefficients{
            .inv_freqs = std::move(freqs),
            .rotary_output_scale = algorithm.rotary_output_scale,
    };
}

} // namespace

Status ValidateRoPEFreqParams(const RoPEParams& params) {
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
                    return Status::InvalidArgument("RoPE Dynamic NTK requires"
                                                   " positive original context and rotary_dim > 2");
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
    const double context_ratio = seq_len / static_cast<double>(original_context_len);
    const double ntk_scale = 1.0 + factor * (context_ratio - 1.0);
    const double exponent = static_cast<double>(rotary_dim) / static_cast<double>(rotary_dim - 2);
    const double base = theta * std::pow(ntk_scale, exponent);
    if (!IsFinitePositive(base)) {
        return Status::Overflow("RoPE Dynamic NTK base frequency is not finite");
    }
    return base;
}

StatusOr<RoPERotationCoefficients> ResolveStaticRoPERotationCoefficients(const RoPEParams& params) {
    AM_RETURN_IF_ERROR(ValidateRoPEFreqParams(params));
    auto visitor = overloaded{
            [&](const StandardRoPE&) -> StatusOr<RoPERotationCoefficients> {
                AM_ASSIGN_OR_RETURN(auto frequencies,
                                    MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
                return RoPERotationCoefficients{.inv_freqs = std::move(frequencies)};
            },
            [&](const LinearRoPE& algorithm) -> StatusOr<RoPERotationCoefficients> {
                AM_ASSIGN_OR_RETURN(auto frequencies,
                                    MakeBaseFreqs(params.theta, EffectiveRoPERotaryDim(params)));
                return RoPERotationCoefficients{
                        .inv_freqs = std::move(frequencies),
                        .position_divisor = algorithm.factor,
                };
            },
            [&](const YarnRoPE& algorithm) -> StatusOr<RoPERotationCoefficients> {
                return ResolveYarn(params, algorithm);
            },
            [&](const Llama3RoPE& algorithm) -> StatusOr<RoPERotationCoefficients> {
                return ResolveLlama3(params, algorithm);
            },
            [](const DynamicNtkRoPE&) -> StatusOr<RoPERotationCoefficients> {
                return Status::InvalidArgument(
                        "RoPE algorithm requires execution-time sequence length");
            },
            [](const LongRoPE&) -> StatusOr<RoPERotationCoefficients> {
                return Status::InvalidArgument(
                        "RoPE algorithm requires execution-time sequence length");
            }};
    return std::visit(visitor, params.algorithm);
}

StatusOr<RoPERotationCoefficients> ResolveDynamicRoPERotationCoefficients(
        const RoPEParams& params,
        int64_t effective_seq_len) {
    AM_RETURN_IF_ERROR(ValidateRoPEFreqParams(params));
    if (!IsDynamicRoPE(params.algorithm)) {
        return Status::InvalidArgument(
                "RoPE algorithm does not require an execution-time sequence length");
    }

    if (effective_seq_len <= 0) {
        return Status::InvalidArgument("RoPE effective sequence length must be positive");
    }

    if (const auto* dynamic_ntk = std::get_if<DynamicNtkRoPE>(&params.algorithm)) {
        return ResolveDynamicNtk(params, *dynamic_ntk, effective_seq_len);
    }
    return ResolveLongRope(params, std::get<LongRoPE>(params.algorithm), effective_seq_len);
}

} // namespace aethermind
