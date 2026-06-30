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
      input_overrides_(graph.GetNodes().size(), std::nullopt),
      node_replacements_(graph.GetNodes().size(), std::nullopt) {}

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

    // Validate: all input/output IDs must reference existing values in the original graph
    for (const GraphNode& replacement: replacement_nodes) {
        for (GraphValueId input: replacement.inputs) {
            AM_RETURN_IF_ERROR(CheckValueId(input));
        }
        for (GraphValueId output: replacement.outputs) {
            AM_RETURN_IF_ERROR(CheckValueId(output));
        }
    }

    removed_nodes_[node.index] = true;
    node_replacements_[node.index] = replacement_nodes;
    return Status::Ok();
}

Status GraphRewriteSession::RemoveNode(GraphNodeId node) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    removed_nodes_[node.index] = true;
    return Status::Ok();
}

Status GraphRewriteSession::RedirectInput(GraphNodeId node, size_t input_index,
                                          GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    AM_RETURN_IF_ERROR(CheckValueId(new_value));
    const GraphNode& original = graph_.GetNode(node);
    if (input_index >= original.inputs.size()) {
        return Status::InvalidArgument(
                "GraphRewriteSession::RedirectInput input index out of range");
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
    // Detect replacement cycle: if new_value's resolution chain already
    // reaches old_value, setting old_value -> new_value would close a cycle.
    // Without this check, GetResolvedValue would iterate the cycle up to
    // value_replacements_.size() times and silently return an arbitrary
    // value along the cycle instead of a stable terminal value.
    GraphValueId cur = new_value;
    for (size_t depth = 0; depth < value_replacements_.size(); ++depth) {
        if (cur.index >= value_replacements_.size()) {
            break;
        }
        const std::optional<GraphValueId>& next = value_replacements_[cur.index];
        if (!next.has_value()) {
            break;
        }
        cur = *next;
        if (cur == old_value) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::ReplaceValue would create a replacement cycle");
        }
    }
    value_replacements_[old_value.index] = new_value;
    return Status::Ok();
}

GraphValueId GraphRewriteSession::GetResolvedValue(GraphValueId value) const {
    GraphValueId cur = value;
    for (size_t depth = 0; depth < value_replacements_.size(); ++depth) {
        if (cur.index >= value_replacements_.size()) {
            return cur;
        }

        const std::optional<GraphValueId>& next = value_replacements_[cur.index];
        if (!next.has_value()) {
            return cur;
        }
        cur = *next;
    }
    return cur;
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
    for (const auto& input_override: input_overrides_) {
        if (!input_override.has_value()) {
            continue;
        }

        for (GraphValueId input: *input_override) {
            AM_RETURN_IF_ERROR(CheckValueId(input));
        }
    }

    for (const auto& replacement: value_replacements_) {
        if (replacement.has_value()) {
            AM_RETURN_IF_ERROR(CheckValueId(*replacement));
        }
    }
    return Status::Ok();
}

StatusOr<ModelGraph> GraphRewriteSession::Commit() const {
    AM_RETURN_IF_ERROR(ValidateEdits());

    ModelGraph committed(graph_.GetConfig());
    std::vector<std::optional<GraphValueId>> value_map(
            graph_.GetValues().size(), std::nullopt);

    const std::span<const GraphValue> values = graph_.GetValues();
    for (size_t i = 0; i < values.size(); ++i) {
        const GraphValueId old_id{.index = static_cast<uint32_t>(i)};
        const GraphValue& value = values[i];
        if (value.producer.has_value()) {
            continue;
        }

        if (std::get_if<ModelInputValue>(&value.payload)) {
            const std::optional<std::string> input_name = FindInputName(graph_, old_id);
            if (!input_name.has_value()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::Commit model input name not found");
            }
            value_map[i] = committed.AddInput(value.spec, *input_name);
        } else if (const auto* weight = std::get_if<WeightValue>(&value.payload)) {
            value_map[i] = committed.AddWeight(value.spec, weight->binding, value.debug_name);
        } else if (const auto* constant = std::get_if<ConstantValue>(&value.payload)) {
            value_map[i] = committed.AddConstant(value.spec, constant->binding, value.debug_name);
        } else if (const auto* state = std::get_if<StateValue>(&value.payload)) {
            value_map[i] = committed.AddState(value.spec, state->binding, value.debug_name);
        } else if (std::holds_alternative<std::monostate>(value.payload)) {
            // External values must be input, weight, constant, or state.
            // A monostate payload indicates an uninitialized value: the source
            // graph is not a valid snapshot and cannot be committed.
            return Status::InvalidArgument(
                    "GraphRewriteSession::Commit external value has unspecified "
                    "(monostate) payload; ModelGraph values must be input, "
                    "weight, constant, or state");
        } else {
            return Status::InvalidArgument(
                    "GraphRewriteSession::Commit external value has unsupported "
                    "payload variant");
        }

        committed.SetQuantization(*value_map[i], value.quantization);
    }

    StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());
    for (GraphNodeId old_node_id: *order) {
        if (removed_nodes_[old_node_id.index]) {
            const auto& replacements = node_replacements_[old_node_id.index];
            if (!replacements.has_value()) {
                continue;
            }

            // Add replacement nodes in caller-provided order
            for (const GraphNode& replacement: *replacements) {
                std::vector<GraphValueId> new_inputs;
                new_inputs.reserve(replacement.inputs.size());
                for (GraphValueId old_input: replacement.inputs) {
                    const GraphValueId resolved_input = GetResolvedValue(old_input);
                    StatusOr<GraphValueId> mapped_input = MapResolvedValue(resolved_input, value_map);
                    AM_RETURN_IF_ERROR(mapped_input.status());
                    new_inputs.push_back(*mapped_input);
                }

                // Look up output specs/payloads from the original graph values
                // referenced by the replacement node's outputs field
                std::vector<NodeOutputDesc> output_descs;
                output_descs.reserve(replacement.outputs.size());
                for (GraphValueId output_id: replacement.outputs) {
                    const GraphValue& old_value = graph_.GetValue(output_id);
                    output_descs.push_back(NodeOutputDesc{
                            .spec = old_value.spec,
                            .payload = old_value.payload,
                            .quantization = old_value.quantization,
                            .debug_name = old_value.debug_name,
                    });
                }

                const AddedNode added = committed.AddNode(
                        replacement.op_type,
                        replacement.decoder_layer_index,
                        std::move(new_inputs),
                        std::move(output_descs),
                        replacement.op_params,
                        replacement.attrs,
                        replacement.debug_name);

                // Map old value IDs to new output IDs so downstream consumers
                // of the replaced node's outputs are redirected to the replacement's outputs
                for (size_t i = 0; i < replacement.outputs.size(); ++i) {
                    value_map[replacement.outputs[i].index] = added.outputs[i];
                }
            }
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

        std::vector<NodeOutputDesc> output_descs;
        output_descs.reserve(view->outputs.size());
        for (GraphValueId old_output: view->outputs) {
            const GraphValue& old_value = graph_.GetValue(old_output);
            output_descs.push_back(NodeOutputDesc{
                    .spec = old_value.spec,
                    .payload = old_value.payload,
                    .quantization = old_value.quantization,
                    .debug_name = old_value.debug_name,
            });
        }

        const AddedNode added = committed.AddNode(
                view->op_type,
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

    for (const auto& output: graph_.GetOutputs()) {
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
