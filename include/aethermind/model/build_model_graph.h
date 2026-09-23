#ifndef AETHERMIND_MODEL_BUILD_MODEL_GRAPH_H
#define AETHERMIND_MODEL_BUILD_MODEL_GRAPH_H

/// @file build_model_graph.h
/// @brief Single dispatch entry for HF → semantic graph conversion.

#include "aethermind/base/status.h"
#include "aethermind/graph/graph.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

/// @brief Converts an HF config/weights pair into a semantic ModelGraph.
///
/// Identifies the model family via ParseModelArchitecture and dispatches to the
/// matching per-family graph builder. Unsupported families are rejected here,
/// so callers (e.g. ModelCompiler) never touch HF family spellings.
StatusOr<ModelGraph> BuildModelGraph(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights);

} // namespace aethermind

#endif // AETHERMIND_MODEL_BUILD_MODEL_GRAPH_H