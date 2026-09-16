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
/// Registered in the default O2 pipeline (after SiluMulFusionPass, before
/// DeadCodeEliminationPass). CPU provides FP32 reference descriptors for both
/// plain and `cpu_identity` packed weights, so a fused node remains one
/// executable kernel step. The `enable_fused_add_rms_norm` PassContext flag
/// (default true) gates the pass at runtime.
class AddRmsNormFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

} // namespace aethermind

#endif
