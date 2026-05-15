#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_config.h"
#include "aethermind/model/raw_weight.h"

namespace aethermind {

class HfModelValidator {
public:
    static Status ValidateConfig(const ModelConfig& config);
    static Status ValidateWeightSet(const ModelConfig& config, const RawWeightTable& weights);
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
