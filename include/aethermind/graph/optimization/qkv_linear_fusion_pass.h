#ifndef AETHERMIND_GRAPH_OPTIMIZATION_QKV_LINEAR_FUSION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_QKV_LINEAR_FUSION_PASS_H

/// @file qkv_linear_fusion_pass.h
/// @brief Q, K, V linear-projection fusion optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Replaces three compatible attention Q/K/V Linear projections with
/// one QkvLinear projection and a fixed composite QKV weight recipe.
///
/// This is deliberately a standalone pass. Registration in a production
/// optimization pipeline remains an explicit model/runtime integration choice,
/// because the pass changes the logical weight recipe consumed by lowering.
class QkvLinearFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    Status Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept override;
};

}// namespace aethermind

#endif
