#include "aethermind/model/loaded_model.h"

#include <utility>

namespace aethermind {

LoadedModel::LoadedModel(HfModelConfig config, ResolvedModelWeights resolved_weights) noexcept
    : LoadedModel(std::move(config), std::move(resolved_weights), std::nullopt) {}

LoadedModel::LoadedModel(HfModelConfig config,
                         ResolvedModelWeights resolved_weights,
                         std::optional<ModelWeightNormalizationUsage> normalization_usage) noexcept
    : config_(std::move(config)),
      resolved_weights_(std::move(resolved_weights)),
      normalization_usage_(normalization_usage) {}

const HfModelConfig& LoadedModel::GetConfig() const noexcept {
    return config_;
}

const ResolvedModelWeights& LoadedModel::GetResolvedWeights() const noexcept {
    return resolved_weights_;
}

const std::optional<ModelWeightNormalizationUsage>& LoadedModel::GetWeightNormalizationUsage() const noexcept {
    return normalization_usage_;
}

} // namespace aethermind
