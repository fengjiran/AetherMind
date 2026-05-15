#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/raw_weight.h"

namespace aethermind {

class HfModelValidator {
public:
    static Status ValidateConfig(const HfModelConfig& config);
    static Status ValidateWeightSet(const HfModelConfig& config, const RawWeightTable& weights);
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_FORMATS_HF_HF_MODEL_VALIDATOR_H
