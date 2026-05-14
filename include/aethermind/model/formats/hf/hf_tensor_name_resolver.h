#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_TENSOR_NAME_RESOLVER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_TENSOR_NAME_RESOLVER_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_config.h"
#include "aethermind/model/resolved_tensor_index.h"

namespace aethermind {

namespace hf {

StatusOr<ResolvedTensorIndex> Resolve(const ModelConfig& config,
                                     const RawTensorTable& tensors);

}// namespace hf

}// namespace aethermind

#endif
