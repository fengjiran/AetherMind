#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_WEIGHT_RESOLVER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_WEIGHT_RESOLVER_H

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

namespace hf {

StatusOr<ResolvedModelWeights> ResolveWeights(const HfModelConfig& config,
                                          const RawWeightTable& weights);

}// namespace hf

}// namespace aethermind

#endif
