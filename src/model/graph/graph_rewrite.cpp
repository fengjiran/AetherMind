#include "aethermind/model/graph/graph_rewrite.h"
#include "utils/variant_utils.h"

#include <variant>

namespace aethermind {
namespace {

std::optional<std::string> FindInputName(const ModelGraph& graph, GraphValueId value) {
    for (const auto& input: graph.GetInputs()) {
        if (input.value == value) {
            return input.name;
        }
    }
    return std::nullopt;
}

StatusOr<GraphValueId> MapResolvedValue(GraphValueId old_value,
                                        const std::vector<std::optional<GraphValueId>>& value_map) {
    if (old_value.index >= value_map.size() || !value_map[old_value.index].has_value()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: value cannot be mapped during commit");
    }
    return *value_map[old_value.index];
}

}// namespace

GraphRewriteSession::GraphRewriteSession(const ModelGraph& graph)
    : graph_(graph),
      removed_nodes_(graph.GetNodes().size(), false),
      value_replacements_(graph.GetValues().size(), std::nullopt),
      input_overrides_(graph.GetNodes().size(), std::nullopt) {}

Status GraphRewriteSession::Apply(std::span<const GraphMutation> mutations) {
    for (const GraphMutation& mutation: mutations) {
        auto visitor = overloaded{
                [&](const ReplaceNodeCmd& replace) {
                    return ReplaceNode(replace.old_node, replace.replacement_nodes);
                },
                [&](const RemoveNodeCmd& remove) {
                    return RemoveNode(remove.node);
                },
                [&](const RedirectInputCmd& redirect) {
                    return RedirectInput(redirect.node, redirect.input_index, redirect.new_value);
                },
                [&](const ReplaceValueCmd& replace) {
                    return ReplaceValue(replace.old_value, replace.new_value);
                },
        };
        AM_RETURN_IF_ERROR(std::visit(visitor, mutation));
    }
    return Status::Ok();
}

Status GraphRewriteSession::ReplaceNode(GraphNodeId node,
                                        const std::vector<GraphNode>& replacement_nodes) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    if (replacement_nodes.empty()) {
        return RemoveNode(node);
    }
    return Status::Unimplemented(
            "GraphRewriteSession::ReplaceNode is not implemented in the minimal M4.3 session");
}

Status GraphRewriteSession::RemoveNode(GraphNodeId node) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    removed_nodes_[node.index] = true;
    return Status::Ok();
}

Status GraphRewriteSession::RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    AM_RETURN_IF_ERROR(CheckValueId(new_value));
    const GraphNode& original = graph_.GetNode(node);
    if (input_index >= original.inputs.size()) {
        return Status::InvalidArgument("GraphRewriteSession::RedirectInput input index out of range");
    }

    if (!input_overrides_[node.index].has_value()) {
        input_overrides_[node.index] = original.inputs;
    }
    (*input_overrides_[node.index])[input_index] = new_value;
    return Status::Ok();
}

Status GraphRewriteSession::ReplaceValue(GraphValueId old_value, GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckValueId(old_value));
    AM_RETURN_IF_ERROR(CheckValueId(new_value));
    if (old_value == new_value) {
        return Status::Ok();
    }
    value_replacements_[old_value.index] = new_value;
    return Status::Ok();
}

Status GraphRewriteSession::ReplaceAllUses(GraphValueId old_value, GraphValueId new_value) {
    return ReplaceValue(old_value, new_value);
}

GraphValueId GraphRewriteSession::GetResolvedValue(GraphValueId value) const {
    GraphValueId current = value;
    for (size_t depth = 0; depth < value_replacements_.size(); ++depth) {
        if (current.index >= value_replacements_.size()) {
            return current;
        }
        const std::optional<GraphValueId>& next = value_replacements_[current.index];
        if (!next.has_value()) {
            return current;
        }
        current = *next;
    }
    return current;
}

StatusOr<GraphNodeView> GraphRewriteSession::GetNodeView(GraphNodeId node) const {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    if (removed_nodes_[node.index]) {
        return Status::NotFound("GraphRewriteSession::GetNodeView node was removed");
    }

    const GraphNode& original = graph_.GetNode(node);
    GraphNodeView view{
            .node = node,
            .op_type = original.op_type,
            .decoder_layer_index = original.decoder_layer_index,
            .inputs = CurrentInputs(node),
            .outputs = original.outputs,
            .attrs = original.attrs,
            .op_params = original.op_params,
            .debug_name = original.debug_name,
    };
    for (GraphValueId& input: view.inputs) {
        input = GetResolvedValue(input);
    }
    return view;
}

Status GraphRewriteSession::ValidateEdits() const {
    for (size_t node_index = 0; node_index < input_overrides_.size(); ++node_index) {
        if (!input_overrides_[node_index].has_value()) {
            continue;
        }
        for (GraphValueId input: *input_overrides_[node_index]) {
            AM_RETURN_IF_ERROR(CheckValueId(input));
        }
    }
    for (const std::optional<GraphValueId>& replacement: value_replacements_) {
        if (replacement.has_value()) {
            AM_RETURN_IF_ERROR(CheckValueId(*replacement));
        }
    }
    return Status::Ok();
}

StatusOr<ModelGraph> GraphRewriteSession::Commit() const {
    AM_RETURN_IF_ERROR(ValidateEdits());

    ModelGraph committed(graph_.GetConfig());
    std::vector<std::optional<GraphValueId>> value_map(graph_.GetValues().size(), std::nullopt);

    const std::span<const GraphValue> values = graph_.GetValues();
    for (size_t i = 0; i < values.size(); ++i) {
        const GraphValueId old_id{.index = static_cast<uint32_t>(i)};
        const GraphValue& value = values[i];
        if (value.producer.has_value()) {
            continue;
        }

        if (const auto* model_input = std::get_if<ModelInputValue>(&value.payload)) {
            (void) model_input;
            const std::optional<std::string> input_name = FindInputName(graph_, old_id);
            if (!input_name.has_value()) {
                return Status::InvalidArgument("GraphRewriteSession::Commit model input name not found");
            }
            value_map[i] = committed.AddInput(value.spec, *input_name);
        } else if (const auto* weight = std::get_if<WeightValue>(&value.payload)) {
            value_map[i] = committed.AddWeight(value.spec, weight->binding, value.debug_name);
        } else if (const auto* state = std::get_if<StateValue>(&value.payload)) {
            value_map[i] = committed.AddState(value.spec, state->binding, value.debug_name);
        } else {
            return Status::InvalidArgument("GraphRewriteSession::Commit external value has invalid payload");
        }
    }

    StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());
    for (GraphNodeId old_node_id: *order) {
        if (removed_nodes_[old_node_id.index]) {
            continue;
        }

        StatusOr<GraphNodeView> view = GetNodeView(old_node_id);
        AM_RETURN_IF_ERROR(view.status());

        std::vector<GraphValueId> new_inputs;
        new_inputs.reserve(view->inputs.size());
        for (GraphValueId old_input: view->inputs) {
            const GraphValueId resolved_input = GetResolvedValue(old_input);
            StatusOr<GraphValueId> mapped_input = MapResolvedValue(resolved_input, value_map);
            AM_RETURN_IF_ERROR(mapped_input.status());
            new_inputs.push_back(*mapped_input);
        }

        std::vector<ModelGraph::NodeOutputDesc> output_descs;
        output_descs.reserve(view->outputs.size());
        for (GraphValueId old_output: view->outputs) {
            const GraphValue& old_value = graph_.GetValue(old_output);
            output_descs.push_back(ModelGraph::NodeOutputDesc{
                    .spec = old_value.spec,
                    .payload = old_value.payload,
                    .debug_name = old_value.debug_name,
            });
        }

        const ModelGraph::AddedNode added = committed.AddNode(view->op_type,
                                                              view->decoder_layer_index,
                                                              std::move(new_inputs),
                                                              std::move(output_descs),
                                                              view->op_params,
                                                              view->attrs,
                                                              view->debug_name);
        for (size_t i = 0; i < view->outputs.size(); ++i) {
            value_map[view->outputs[i].index] = added.outputs[i];
        }
    }

    for (const ModelGraph::Output& output: graph_.GetOutputs()) {
        const GraphValueId resolved_output = GetResolvedValue(output.value);
        StatusOr<GraphValueId> mapped_output = MapResolvedValue(resolved_output, value_map);
        AM_RETURN_IF_ERROR(mapped_output.status());
        committed.MarkOutput(*mapped_output, output.name);
    }

    AM_RETURN_IF_ERROR(committed.Validate());
    return committed;
}

Status GraphRewriteSession::CheckNodeId(GraphNodeId node) const {
    if (node.index >= graph_.GetNodes().size()) {
        return Status::InvalidArgument("GraphRewriteSession: node id out of range");
    }
    return Status::Ok();
}

Status GraphRewriteSession::CheckValueId(GraphValueId value) const {
    if (value.index >= graph_.GetValues().size()) {
        return Status::InvalidArgument("GraphRewriteSession: value id out of range");
    }
    return Status::Ok();
}

const std::vector<GraphValueId>& GraphRewriteSession::CurrentInputs(GraphNodeId node) const {
    if (input_overrides_[node.index].has_value()) {
        return *input_overrides_[node.index];
    }
    return graph_.GetNode(node).inputs;
}

}// namespace aethermind
