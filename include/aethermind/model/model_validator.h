#ifndef AETHERMIND_MODEL_MODEL_VALIDATOR_H
#define AETHERMIND_MODEL_MODEL_VALIDATOR_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_config.h"
#include "aethermind/model/resolved_tensor_index.h"

namespace aethermind {

class ModelValidator {
public:
    static Status ValidateConfig(const ModelConfig& config);
    static Status ValidateTensorSet(const ModelConfig& config, const RawTensorTable& tensors);
};

}// namespace aethermind

#endif// AETHERMIND_MODEL_MODEL_VALIDATOR_H
