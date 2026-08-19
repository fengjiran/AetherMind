#ifndef AETHERMIND_COMPILER_SEMANTIC_OPTIMIZATION_PIPELINE_H
#define AETHERMIND_COMPILER_SEMANTIC_OPTIMIZATION_PIPELINE_H

/// @file semantic_optimization_pipeline.h
/// @brief Compiler-owned composition of backend-independent graph passes.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// Runs the compiler's default semantic optimization pipeline. Individual
/// passes and the pass framework remain graph-owned; only pass ordering and
/// compilation-stage composition live in compiler.
AM_NODISCARD StatusOr<ModelGraph> OptimizeModelGraph(
        const ModelGraph& graph,
        PassContext context = {});

}// namespace aethermind

#endif
