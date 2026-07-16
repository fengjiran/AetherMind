#ifndef AETHERMIND_MODEL_GRAPH_OPTIMIZATION_SILU_MUL_FUSION_PASS_H
#define AETHERMIND_MODEL_GRAPH_OPTIMIZATION_SILU_MUL_FUSION_PASS_H

#include "aethermind/model/graph/optimization/graph_pass_manager.h"

namespace aethermind {

class SiluMulFusionPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext& ctx) override;
};

}// namespace aethermind

#endif
