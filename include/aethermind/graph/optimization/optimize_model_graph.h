#ifndef AETHERMIND_GRAPH_OPTIMIZATION_OPTIMIZE_MODEL_GRAPH_H
#define AETHERMIND_GRAPH_OPTIMIZATION_OPTIMIZE_MODEL_GRAPH_H

/// @file optimize_model_graph.h
/// @brief Optimization entry point for the semantic model graph.
///
/// The graph-to-executable flow is two explicit stages that callers compose
/// directly: OptimizeModelGraph (semantic graph rewrites, declared here)
/// followed by LowerModelGraph (graph → LoweredGraph, declared in
/// graph_lowering.h).
#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Applies the default optimization pipeline for the requested opt_level.
///
/// Pass selection is deterministic and driven solely by opt_level:
///   O0 — no passes
///   O1 — ConstantFoldingPass → DeadCodeEliminationPass
///   O2+ — ConstantFoldingPass → QkvLinearFusionPass → GateUpLinearFusionPass
///         → SiluMulFusionPass → AddRmsNormFusionPass → DeadCodeEliminationPass
///
/// @param graph Source semantic graph. Never mutated.
/// @param context Optimization context. Feature flags, checkpoint_every, and
/// const_eval_policy are forwarded to every pass unchanged.
/// @return Optimized ModelGraph on success, or an error status describing the
/// first pass failure.
StatusOr<ModelGraph> OptimizeModelGraph(
        const ModelGraph& graph,
        PassContext context = {});

}// namespace aethermind

#endif
