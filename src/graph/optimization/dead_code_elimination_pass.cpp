#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/operators/operator_schema.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace aethermind {
namespace {

bool IsDceRemovableOp(OpType op_type) {
    const auto schema = GetOperatorSchema(op_type);
    if (!schema.ok()) {
        return false;
    }
    return !schema->traits.has_side_effects && !HasStatefulOutput(*schema);
}

StatusOr<bool> AreAllOutputsDead(const GraphRewriteSession& session,
                                 const GraphNodeView& node,
                                 const std::vector<GraphValueId>& output_terminals) {
    for (const auto output: node.outputs) {
        // When an earlier pass (e.g. constant folding) replaced this output
        // with a different value, the original output is dead from this
        // producer's perspective. Consumers are now attached to the
        // replacement value and graph outputs resolve through the chain to
        // its terminal, so this node is removable; whether the terminal's own
        // producer is removable is decided when that node is visited.
        if (session.GetResolvedValue(output) != output) {
            continue;
        }

        // Liveness of an unreplaced output: Commit's MarkCommittedOutputs
        // marks the terminal of every graph output in the committed graph, so
        // a node whose output IS such a terminal must survive even when the
        // output is not directly marked as graph output (a replacement chain
        // may start at a marked output and end here). Removing the terminal's
        // producer would leave the output unmappable during commit.
        if (std::ranges::find(output_terminals, output) != output_terminals.end()) {
            return false;
        }

        // HasLiveConsumers returns InvalidArgument if the source graph
        // contains a cycle; propagate it so the failure is reported instead
        // of silently keeping or dropping the node.
        AM_ASSIGN_OR_RETURN(bool has_consumers, session.HasLiveConsumers(output));
        if (has_consumers) {
            return false;
        }
    }
    return true;
}

Status RemoveDeadNodesOnce(GraphRewriteSession& session,
                           const std::vector<GraphValueId>& output_terminals,
                           bool& changed) {
    StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());

    for (std::size_t i = order->size(); i > 0U; --i) {
        const GraphNodeId node_id = (*order)[i - 1U];
        StatusOr<GraphNodeView> node = session.GetNodeView(node_id);
        AM_RETURN_IF_ERROR(node.status());
        AM_ASSIGN_OR_RETURN(bool all_outputs_dead,
                            AreAllOutputsDead(session, *node, output_terminals));
        if (!IsDceRemovableOp(node->op_type) || !all_outputs_dead) {
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

Status DeadCodeEliminationPass::Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept {
    if (!ctx.enable_dce) {
        return Status::Ok();
    }

    // Resolved terminals of the graph outputs. DCE never adds replacements,
    // so the terminal set is stable across the fixed-point iterations.
    const std::vector<GraphValueId> output_terminals = session.GetResolvedGraphOutputs();

    bool changed = true;
    while (changed) {
        changed = false;
        AM_RETURN_IF_ERROR(RemoveDeadNodesOnce(session, output_terminals, changed));
    }
    return Status::Ok();
}

}// namespace aethermind
