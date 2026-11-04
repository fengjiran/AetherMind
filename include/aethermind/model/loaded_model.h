#ifndef AETHERMIND_MODEL_LOADED_MODEL_H
#define AETHERMIND_MODEL_LOADED_MODEL_H

/// @file loaded_model.h
/// @brief Ownership boundary for a validated Hugging Face model before compilation.
/// @see docs/designs/model/01-model-loader.md

#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight/raw_weight_fp32_conversion.h"

#include <optional>

namespace aethermind {

/// @brief Loaded model state: config plus resolved logical weights.
///
/// A model that has completed HF I/O, validation, and logical-weight
/// resolution. LoadedModel deliberately contains no backend artifacts. Its
/// ResolvedModelWeights retain the shared backing storage for RawWeightView
/// instances until graph-driven materialization consumes them in a later
/// compilation stage.
class LoadedModel {
public:
    LoadedModel(HfModelConfig config, ResolvedModelWeights resolved_weights) noexcept;
    LoadedModel(HfModelConfig config,
                ResolvedModelWeights resolved_weights,
                const std::optional<RawWeightFp32ConversionUsage>& fp32_conversion_usage) noexcept;

    AM_NODISCARD const HfModelConfig& GetConfig() const noexcept;
    AM_NODISCARD const ResolvedModelWeights& GetResolvedWeights() const noexcept;

    /// @brief Observes conversion accounting without retaining source backing.
    AM_NODISCARD const std::optional<RawWeightFp32ConversionUsage>&
    GetWeightFp32ConversionUsage() const noexcept;

private:
    HfModelConfig config_{};
    ResolvedModelWeights resolved_weights_{};
    std::optional<RawWeightFp32ConversionUsage> fp32_conversion_usage_{};
};

} // namespace aethermind

#endif
