#ifndef AETHERMIND_GRAPH_OPTIMIZATION_GATE_UP_LINEAR_FUSION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_GATE_UP_LINEAR_FUSION_PASS_H

/// @file gate_up_linear_fusion_pass.h
/// @brief MLP gate/up linear-projection fusion optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Replaces compatible MLP gate/up Linear projections with one
/// GateUpLinear projection and a fixed Gate-Up composite weight recipe.
///
/// The pass changes graph semantics and logical weight binding, never
/// materializes or prepacks a physical fused weight. Packed-weight lowering
/// materializes the Gate-Up recipe for kernels that require it.
class GateUpLinearFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

} // namespace aethermind

#endif
