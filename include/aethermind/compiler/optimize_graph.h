#ifndef AETHERMIND_COMPILER_OPTIMIZE_GRAPH_H
#define AETHERMIND_COMPILER_OPTIMIZE_GRAPH_H

/// @file optimize_graph.h
/// @brief Compiler-owned composition of backend-independent graph passes.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// Runs the compiler's default semantic optimization pipeline. Individual
/// passes and the pass framework remain graph-owned; only pass ordering and
/// compilation-stage composition live in compiler.
StatusOr<ModelGraph> OptimizeModelGraph(
        const ModelGraph& graph,
        PassContext context = {});

} // namespace aethermind

#endif
