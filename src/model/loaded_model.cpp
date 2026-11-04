#include "aethermind/model/loaded_model.h"

#include <utility>

namespace aethermind {

LoadedModel::LoadedModel(HfModelConfig config, ResolvedModelWeights resolved_weights) noexcept
    : LoadedModel(std::move(config),
                  std::move(resolved_weights),
                  std::nullopt) {}

LoadedModel::LoadedModel(
        HfModelConfig config,
        ResolvedModelWeights resolved_weights,
        const std::optional<RawWeightFp32ConversionUsage>& fp32_conversion_usage) noexcept
    : config_(std::move(config)),
      resolved_weights_(std::move(resolved_weights)),
      fp32_conversion_usage_(fp32_conversion_usage) {}

const HfModelConfig& LoadedModel::GetConfig() const noexcept {
    return config_;
}

const ResolvedModelWeights& LoadedModel::GetResolvedWeights() const noexcept {
    return resolved_weights_;
}

const std::optional<RawWeightFp32ConversionUsage>&
LoadedModel::GetWeightFp32ConversionUsage() const noexcept {
    return fp32_conversion_usage_;
}

} // namespace aethermind
