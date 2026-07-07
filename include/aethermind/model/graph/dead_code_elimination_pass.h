#ifndef AETHERMIND_MODEL_GRAPH_DEAD_CODE_ELIMINATION_PASS_H
#define AETHERMIND_MODEL_GRAPH_DEAD_CODE_ELIMINATION_PASS_H

#include "aethermind/model/graph/graph_pass_manager.h"

namespace aethermind {

class DeadCodeEliminationPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override;
    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext& ctx) override;
};

}// namespace aethermind

#endif
