#ifndef AETHERMIND_GRAPH_OPTIMIZATION_GATE_UP_LINEAR_FUSION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_GATE_UP_LINEAR_FUSION_PASS_H

/// @file gate_up_linear_fusion_pass.h
/// @brief MLP gate/up linear-projection fusion optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Replaces compatible MLP gate/up Linear projections with one
/// GateUpLinear projection and a fixed Gate-Up composite weight recipe.
///
/// The pass is registered at O2 but disabled by default until execution has a
/// GateUpLinear kernel. It changes graph semantics and logical weight binding,
/// never materializes or prepacks a physical fused weight.
class GateUpLinearFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

}// namespace aethermind

#endif
