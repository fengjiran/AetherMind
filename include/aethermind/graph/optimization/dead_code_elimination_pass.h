#ifndef AETHERMIND_GRAPH_OPTIMIZATION_DEAD_CODE_ELIMINATION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_DEAD_CODE_ELIMINATION_PASS_H

/// @file dead_code_elimination_pass.h
/// @brief Dead code elimination optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Removes DCE-removable graph nodes whose outputs are dead.
///
/// Resolves graph-output terminals once, then repeatedly removes nodes whose
/// unreplaced outputs are neither terminals nor consumed by a live node.
/// After a successful enabled run, requests Commit-time mixed-graph pruning
/// to remove unreachable replacement residue. When `enable_dce` is false,
/// the pass makes no changes and does not request Commit-time pruning.
class DeadCodeEliminationPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

}// namespace aethermind

#endif
