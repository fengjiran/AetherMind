#ifndef AETHERMIND_MODEL_LOADED_MODEL_H
#define AETHERMIND_MODEL_LOADED_MODEL_H

/// @file loaded_model.h
/// @brief Ownership boundary for a validated Hugging Face model before compilation.

#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

/// A model that has completed HF I/O, validation, and logical-weight resolution.
///
/// LoadedModel deliberately contains no backend artifacts. Its ResolvedModelWeights
/// retain the shared backing storage for RawWeightView instances until graph-driven
/// materialization consumes them in a later compilation stage.
class LoadedModel {
public:
    LoadedModel(HfModelConfig config, ResolvedModelWeights resolved_weights) noexcept;

    AM_NODISCARD const HfModelConfig& GetConfig() const noexcept;
    AM_NODISCARD const ResolvedModelWeights& GetResolvedWeights() const noexcept;

private:
    HfModelConfig config_{};
    ResolvedModelWeights resolved_weights_{};
};

}// namespace aethermind

#endif
