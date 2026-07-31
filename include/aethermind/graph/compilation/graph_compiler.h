#ifndef AETHERMIND_GRAPH_COMPILATION_GRAPH_COMPILER_H
#define AETHERMIND_GRAPH_COMPILATION_GRAPH_COMPILER_H

/// @file graph_compiler.h
/// @brief Full graph compilation pipeline: optimization passes followed by lowering.
///
/// Wraps OptimizeModelGraph and LowerModelGraph into a single owning
/// CompiledModelGraph artifact.
#include "aethermind/graph/compilation/graph_lowering.h"
#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Configuration for the full graph compilation pipeline.
///
/// @note Defaults: optimization at O2 (ConstantFolding → SiluMulFusion → DCE),
/// lowering at CPU/scalar/plain/both.
struct GraphCompileConfig {
    PassContext optimization{};
    GraphLoweringConfig lowering{};
};

/// @brief Owning artifact of Optimize → Lower compilation.
///
/// @warning optimized_graph must outlive lowered because LoweredGraph stores
/// GraphValueId references whose metadata (standalone constants, resolved state
/// aliases) lives in the ModelGraph value table.
struct CompiledModelGraph {
    ModelGraph optimized_graph{};
    LoweredGraph lowered{};
};

/// @brief Applies the default optimization pipeline for the requested opt_level.
///
/// Pass selection is deterministic and driven solely by opt_level:
///   O0 — no passes
///   O1 — ConstantFoldingPass → DeadCodeEliminationPass
///   O2+ — ConstantFoldingPass → SiluMulFusionPass → DeadCodeEliminationPass
///
/// @param graph Source semantic graph. Never mutated.
/// @param context Optimization context. Feature flags, checkpoint_every, and
/// const_eval_policy are forwarded to every pass unchanged.
/// @return Optimized ModelGraph on success, or an error status describing the
/// first pass failure.
AM_NODISCARD StatusOr<ModelGraph> OptimizeModelGraph(
        const ModelGraph& graph,
        PassContext context = {});

/// @brief Optimizes and lowers a model graph in strict sequence.
///
/// Steps:
/// 1. OptimizeModelGraph(graph, config.optimization). Returns the error status
///    immediately on failure.
/// 2. LowerModelGraph(optimized, config.lowering). Returns the error status
///    immediately on failure.
/// 3. Moves both artifacts into CompiledModelGraph and returns it.
///
/// @param graph Source semantic graph. Never mutated.
/// @param config Compilation configuration for optimization and lowering.
/// @return CompiledModelGraph on success, or an error status describing the
/// first failure. No fallback to unoptimized graph.
///
/// @note The optimized graph is retained so that GraphValueIds in
/// LoweredGraph::step_bindings remain meaningful; standalone constant graph
/// outputs stay accessible through optimized_graph.
AM_NODISCARD StatusOr<CompiledModelGraph> CompileModelGraph(
        const ModelGraph& graph,
        const GraphCompileConfig& config = {});

}// namespace aethermind

#endif
