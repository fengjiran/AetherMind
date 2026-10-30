#ifndef AETHERMIND_MODEL_BUILD_MODEL_GRAPH_H
#define AETHERMIND_MODEL_BUILD_MODEL_GRAPH_H

/// @file build_model_graph.h
/// @brief Single dispatch entry for HF → semantic graph conversion.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/graph/graph.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/resolved_model_weights.h"

namespace aethermind {

/// @brief Explicit semantic dtypes used when building a model graph.
///
/// Undefined fields preserve the builder's legacy policy: the checkpoint hint
/// (Float32 when absent) supplies the default KV dtype, while activation specs
/// follow operator inference. An explicit activation policy becomes the default
/// KV dtype and must match the embedding output dtype until mixed-output
/// embedding semantics are available. Overrides do not convert weight payloads
/// or alter the source configuration.
struct ModelGraphBuildOptions {
    DataType activation_dtype{};
    DataType kv_dtype{};
};

/// @brief Converts an HF config/weights pair into a semantic ModelGraph.
///
/// Identifies the model family via ParseModelArchitecture and dispatches to the
/// matching per-family graph builder. Unsupported families are rejected here,
/// so callers (e.g. ModelCompiler) never touch HF family spellings.
StatusOr<ModelGraph> BuildModelGraph(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights);

/// @brief Builds a graph with explicit activation/KV dtype policy.
/// @note Unsupported dtype combinations fail graph construction; an override
///       does not imply that a backend implements the resulting graph.
StatusOr<ModelGraph> BuildModelGraph(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights,
                                     const ModelGraphBuildOptions& options);

} // namespace aethermind

#endif // AETHERMIND_MODEL_BUILD_MODEL_GRAPH_H
