#ifndef AETHERMIND_MODEL_GRAPH_COMPILATION_GRAPH_COMPILER_H
#define AETHERMIND_MODEL_GRAPH_COMPILATION_GRAPH_COMPILER_H

#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/model/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// Configuration for the full graph compilation pipeline.
///
/// Defaults: optimization at O2 (ConstantFolding → SiluMulFusion → DCE),
/// lowering at CPU/scalar/plain/both.
struct GraphCompileConfig {
    PassContext optimization{};
    GraphLoweringConfig lowering{};
};

/// Owning artifact of Optimize → Lower compilation.
///
/// optimized_graph must outlive lowered because LoweredGraph stores
/// GraphValueId references whose metadata (standalone constants,
/// resolved state aliases) lives in the ModelGraph value table.
struct CompiledModelGraph {
    ModelGraph optimized_graph{};
    LoweredGraph lowered{};
};

/// Applies the default optimization pipeline for the requested opt_level.
///
/// Pass selection is deterministic and driven solely by opt_level:
///   O0 — no passes
///   O1 — ConstantFoldingPass → DeadCodeEliminationPass
///   O2+ — ConstantFoldingPass → SiluMulFusionPass → DeadCodeEliminationPass
///
/// The source graph is never mutated. Feature flags, checkpoint_every, and
/// const_eval_policy are forwarded to every pass unchanged.
AM_NODISCARD StatusOr<ModelGraph> OptimizeModelGraph(
        const ModelGraph& graph,
        PassContext context = {});

/// Optimizes and lowers a model graph in strict sequence.
///
/// 1. OptimizeModelGraph(graph, config.optimization).
///    Returns the error status immediately on failure.
/// 2. LowerModelGraph(optimized, config.lowering).
///    Returns the error status immediately on failure.
/// 3. Moves both artifacts into CompiledModelGraph and returns it.
///
/// The optimized graph is retained so that GraphValueIds in
/// LoweredGraph::step_bindings remain meaningful; standalone constant
/// graph outputs stay accessible through optimized_graph.
///
/// The source graph is never mutated. No fallback to unoptimized graph.
AM_NODISCARD StatusOr<CompiledModelGraph> CompileModelGraph(
        const ModelGraph& graph,
        const GraphCompileConfig& config = {});

}// namespace aethermind

#endif
