#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_WEIGHT_RESOLVER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_WEIGHT_RESOLVER_H

/// @file hf_weight_resolver.h
/// @brief HF tensor-name resolution into logical model weight views.

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

namespace hf {

/// @brief Resolves raw HF tensors into logical model weight views.
///
/// Applies HF naming conventions (e.g. tied embedding aliases) so that
/// downstream consumers see one logical weight per model role.
///
/// @param config Parsed HF config defining the logical weight schema.
/// @param weights Raw weight table from the safetensors layout.
/// @return Resolved weights, or an error for unexpected or missing tensors.
StatusOr<ResolvedModelWeights> ResolveWeights(const HfModelConfig& config,
                                              const RawWeightTable& weights);

} // namespace hf

} // namespace aethermind

#endif
