#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H

/// @file hf_model_validator.h
/// @brief Three-stage structural validation for HF models and weights.

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/model_validation_options.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

/// @brief Validates HF model structure without touching execution layers.
///
/// Validation is split into stages that mirror the load pipeline:
/// ValidateConfig (config.json semantics), ValidateWeightSet (raw weight
/// table vs. schema), and ValidateResolvedModel (resolved weight integrity).
/// Failures are reported as Status; option toggles relax or tighten policy.
class HfModelValidator {
public:
    /// @brief Validates config.json semantic fields against Phase 1 scope.
    ///
    /// @param config Parsed HF config.
    /// @param options Policy toggles; defaults accept common HF export quirks.
    /// @return InvalidArgument on unsupported or inconsistent configuration.
    static Status ValidateConfig(const HfModelConfig& config,
                                 const ModelValidationOptions& options = {});

    /// @brief Validates the raw weight table against the model schema.
    ///
    /// Checks that every schema tensor is present and that unexpected
    /// tensors comply with the configured strictness policy.
    ///
    /// @param config Parsed HF config defining the expected tensor schema.
    /// @param weights Raw weight table loaded from safetensors files.
    /// @param options Policy toggles for tensor-name strictness.
    /// @return InvalidArgument on schema mismatches.
    static Status ValidateWeightSet(const HfModelConfig& config,
                                    const RawWeightTable& weights,
                                    const ModelValidationOptions& options = {});

    /// @brief Validates resolved weight integrity (tied embeddings, dtypes).
    ///
    /// @param config Parsed HF config.
    /// @param resolved Resolved logical weights.
    /// @param options Policy toggles (e.g. uniform linear dtype).
    /// @return InvalidArgument on integrity violations.
    static Status ValidateResolvedModel(const HfModelConfig& config,
                                        const ResolvedModelWeights& resolved,
                                        const ModelValidationOptions& options = {});
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
