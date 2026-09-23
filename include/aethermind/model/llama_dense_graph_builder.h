#ifndef AETHERMIND_MODEL_LLAMA_DENSE_GRAPH_BUILDER_H
#define AETHERMIND_MODEL_LLAMA_DENSE_GRAPH_BUILDER_H

/// @file llama_dense_graph_builder.h
/// @brief Llama dense HF → semantic graph conversion.

#include "aethermind/base/status.h"
#include "aethermind/graph/graph.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

/// @brief Converts a Llama dense config/weights pair into a semantic ModelGraph.
///
/// Together with the shared RoPE normalization this is the single authority for
/// the HF → semantic graph conversion of dense Llama checkpoints; unsupported
/// HF RoPE variants are rejected without producing a partial graph. Dispatched
/// by BuildModelGraph; direct callers bypass family identification.
StatusOr<ModelGraph> BuildLlamaDense(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights);

} // namespace aethermind

#endif // AETHERMIND_MODEL_LLAMA_DENSE_GRAPH_BUILDER_H