#include "aethermind/model/loaded_model.h"

#include <utility>

namespace aethermind {

LoadedModel::LoadedModel(HfModelConfig config, ResolvedModelWeights resolved_weights) noexcept
    : config_(std::move(config)),
      resolved_weights_(std::move(resolved_weights)) {}

const HfModelConfig& LoadedModel::GetConfig() const noexcept {
    return config_;
}

const ResolvedModelWeights& LoadedModel::GetResolvedWeights() const noexcept {
    return resolved_weights_;
}

}// namespace aethermind
