#ifndef AETHERMIND_GRAPH_OPTIMIZATION_DEAD_CODE_ELIMINATION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_DEAD_CODE_ELIMINATION_PASS_H

/// @file dead_code_elimination_pass.h
/// @brief Dead code elimination optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Removes graph nodes whose outputs have no live consumers and are
/// not graph outputs or side-effect roots.
///
/// Uses GraphRewriteSession liveness queries (IsValueLive, HasLiveConsumers,
/// IsGraphOutput) to decide eligibility, then drops dead nodes via
/// RemoveNode. Side-effecting operators are preserved unconditionally.
class DeadCodeEliminationPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

}// namespace aethermind

#endif
