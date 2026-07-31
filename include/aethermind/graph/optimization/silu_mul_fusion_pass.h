#ifndef AETHERMIND_GRAPH_OPTIMIZATION_SILU_MUL_FUSION_PASS_H
#define AETHERMIND_GRAPH_OPTIMIZATION_SILU_MUL_FUSION_PASS_H

/// @file silu_mul_fusion_pass.h
/// @brief SiLU × Mul fusion optimization pass.

#include "aethermind/graph/optimization/graph_pass_manager.h"

namespace aethermind {

/// @brief Fuses adjacent SiLU and Mul operator nodes into a single fused
/// SiluMul node via subgraph replacement.
///
/// Matches the pattern `Mul(Silu(x), y)` and rewrites it through
/// GraphRewriteSession::ReplaceSubgraph, reducing operator dispatch and
/// intermediate value traffic for the Llama MLP activation path.
class SiluMulFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext& ctx) override;
};

}// namespace aethermind

#endif
