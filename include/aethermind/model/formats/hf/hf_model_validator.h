#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/model_validation_options.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/raw_weight.h"

namespace aethermind {

class HfModelValidator {
public:
    static Status ValidateConfig(const HfModelConfig& config,
                                 const ModelValidationOptions& options = {});

    static Status ValidateWeightSet(const HfModelConfig& config,
                                    const RawWeightTable& weights,
                                    const ModelValidationOptions& options = {});

    static Status ValidateResolvedModel(const HfModelConfig& config,
                                        const ResolvedModelWeights& resolved,
                                        const ModelValidationOptions& options = {});
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
