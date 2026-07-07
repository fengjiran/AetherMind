#include "aethermind/model/graph/dead_code_elimination_pass.h"
#include "aethermind/model/graph/operator_schema.h"

#include <cstddef>
#include <vector>

namespace aethermind {
namespace {

bool IsDceRemovableOp(OpType op_type) {
    const StatusOr<OperatorSchema> schema = GetOperatorSchema(op_type);
    if (!schema.ok()) {
        return false;
    }
    return !schema->traits.has_side_effects && !HasStatefulOutput(*schema);
}

bool AreAllOutputsDead(const GraphRewriteSession& session, const GraphNodeView& node) {
    for (GraphValueId output: node.outputs) {
        if (session.IsGraphOutput(output) || session.HasLiveConsumers(output)) {
            return false;
        }
    }
    return true;
}

Status RemoveDeadNodesOnce(GraphRewriteSession& session, bool& changed) {
    StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());

    for (std::size_t i = order->size(); i > 0U; --i) {
        const GraphNodeId node_id = (*order)[i - 1U];
        StatusOr<GraphNodeView> node = session.GetNodeView(node_id);
        AM_RETURN_IF_ERROR(node.status());
        if (!IsDceRemovableOp(node->op_type) || !AreAllOutputsDead(session, *node)) {
            continue;
        }

        AM_RETURN_IF_ERROR(session.RemoveNode(node_id));
        changed = true;
    }
    return Status::Ok();
}

}// namespace

std::string_view DeadCodeEliminationPass::Name() const noexcept {
    return "DeadCodeEliminationPass";
}

Status DeadCodeEliminationPass::Run(GraphRewriteSession& session, const PassContext& ctx) {
    if (!ctx.enable_dce) {
        return Status::Ok();
    }

    bool changed = true;
    while (changed) {
        changed = false;
        AM_RETURN_IF_ERROR(RemoveDeadNodesOnce(session, changed));
    }
    return Status::Ok();
}

}// namespace aethermind
