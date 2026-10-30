#ifndef AETHERMIND_MODEL_LLAMA_DENSE_GRAPH_BUILDER_H
#define AETHERMIND_MODEL_LLAMA_DENSE_GRAPH_BUILDER_H

/// @file llama_dense_graph_builder.h
/// @brief Llama dense HF → semantic graph conversion.

#include "aethermind/model/build_model_graph.h"

namespace aethermind {

/// @brief Converts a Llama dense config/weights pair into a semantic ModelGraph.
///
/// Together with the shared RoPE normalization this is the single authority for
/// the HF → semantic graph conversion of dense Llama checkpoints; unsupported
/// HF RoPE variants are rejected without producing a partial graph. Dispatched
/// by BuildModelGraph; direct callers bypass family identification.
StatusOr<ModelGraph> BuildLlamaDense(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights);

/// @brief Builds Llama semantics with explicit activation/KV dtypes.
StatusOr<ModelGraph> BuildLlamaDense(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights,
                                     const ModelGraphBuildOptions& options);

} // namespace aethermind

#endif // AETHERMIND_MODEL_LLAMA_DENSE_GRAPH_BUILDER_H
