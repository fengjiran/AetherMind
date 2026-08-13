#ifndef AETHERMIND_GRAPH_OPTIMIZATION_ADD_RMS_NORM_FUSION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_ADD_RMS_NORM_FUSION_PASS_H

/// @file add_rmsnorm_fusion_pass.h
/// @brief Add followed by RMS normalization fusion optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Replaces a compatible `RmsNorm(Add(...), weight)` pair with one
/// AddRmsNorm node.
///
/// The fused node preserves both observable values: output 0 takes over the
/// RmsNorm output, while output 1 takes over the Add output. Keeping the Add
/// value identity allows its other consumers, graph outputs, and later layer
/// norms to remain valid without introducing an activation alias contract.
///
/// This pass is intentionally standalone. It is not registered in the default
/// O2 pipeline because backend support for AddRmsNorm is an explicit runtime
/// integration decision.
class AddRmsNormFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

}// namespace aethermind

#endif
