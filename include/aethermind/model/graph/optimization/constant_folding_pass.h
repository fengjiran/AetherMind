#ifndef AETHERMIND_MODEL_GRAPH_OPTIMIZATION_CONSTANT_FOLDING_PASS_H
#define AETHERMIND_MODEL_GRAPH_OPTIMIZATION_CONSTANT_FOLDING_PASS_H

#include "aethermind/model/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// Op-agnostic constant folding pass that materializes compile-time-evaluable
/// subgraphs into static constants.
///
/// The pass iterates the graph in topological order, identifies nodes whose
/// operator schema is marked `compile_time_evaluable` and whose inputs are all
/// inline constants, then delegates to the corresponding `ConstEvaluator` via
/// the Plan + Evaluate contract. Nodes that cannot be folded (evaluator returns
/// Unimplemented, or exceeds the ConstEvalPolicy budget) are silently skipped.
///
/// The pass never allocates output memory itself — it uses the evaluator's
/// Plan() to determine layout and CountBytes for sizing, then allocates and
/// evaluates. Folded constants are committed to the session via AddConstant
/// and ReplaceValue, completing the rewrite.
class ConstantFoldingPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext& ctx) override;
};

}// namespace aethermind

#endif
