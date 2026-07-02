#include "aethermind/model/graph/graph_rewrite.h"
#include "utils/variant_utils.h"

#include <array>
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

NodeOutputDesc MakeOutputDescFromValue(const GraphValue& value) {
    return NodeOutputDesc{
            .spec = value.spec,
            .payload = value.payload,
            .quantization = value.quantization,
            .debug_name = value.debug_name,
    };
}

}// namespace

GraphRewriteSession::GraphRewriteSession(const ModelGraph& graph)
    : graph_(graph),
      node_rewrites_(graph.GetNodes().size()),
      value_replacements_(graph.GetValues().size(), std::nullopt),
      resolved_value_cache_(graph.GetValues().size(), std::nullopt),
      input_overrides_(graph.GetNodes().size(), std::nullopt) {}

Status GraphRewriteSession::Apply(std::span<const GraphMutation> mutations) {
    auto visitor = overloaded{
            [this](const ReplaceNodeCmd& replace) {
                return ReplaceNode(replace.old_node, replace.replacement_nodes);
            },
            [this](const ReplaceSubgraphCmd& replace) {
                return ReplaceSubgraph(replace.old_nodes, replace.replacement_nodes);
            },
            [this](const RemoveNodeCmd& remove) {
                return RemoveNode(remove.node);
            },
            [this](const RedirectInputCmd& redirect) {
                return RedirectInput(redirect.node, redirect.input_index, redirect.new_value);
            },
            [this](const ReplaceValueCmd& replace) {
                return ReplaceValue(replace.old_value, replace.new_value);
            },
    };

    for (const GraphMutation& mutation: mutations) {
        AM_RETURN_IF_ERROR(std::visit(visitor, mutation));
    }
    return Status::Ok();
}

Status GraphRewriteSession::RemoveNode(GraphNodeId node) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    NodeRewriteEntry& rewrite = node_rewrites_[node.index];
    if (rewrite.subgraph_rewrite.has_value()) {
        ClearSubgraphRewrite(*rewrite.subgraph_rewrite);
    }
    rewrite.kind = NodeRewriteKind::kRemove;
    rewrite.subgraph_rewrite.reset();
    return Status::Ok();
}

Status GraphRewriteSession::ReplaceNode(GraphNodeId node,
                                        const std::vector<GraphNode>& replacement_nodes) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));

    std::vector<ReplacementNode> converted;
    converted.reserve(replacement_nodes.size());
    for (const GraphNode& replacement: replacement_nodes) {
        ReplacementNode converted_node{
                .op_type = replacement.op_type,
                .decoder_layer_index = replacement.decoder_layer_index,
                .inputs = replacement.inputs,
                .attrs = replacement.attrs,
                .op_params = replacement.op_params,
                .debug_name = replacement.debug_name,
        };
        converted_node.outputs.reserve(replacement.outputs.size());
        for (GraphValueId output: replacement.outputs) {
            AM_RETURN_IF_ERROR(CheckValueId(output));
            converted_node.outputs.push_back(ReplacementOutput{
                    .desc = MakeOutputDescFromValue(graph_.GetValue(output)),
                    .replaces = output,
            });
        }
        for (GraphValueId input: replacement.inputs) {
            AM_RETURN_IF_ERROR(CheckValueId(input));
        }
        converted.push_back(std::move(converted_node));
    }

    const std::array old_nodes{node};
    return ReplaceSubgraph(old_nodes, converted);
}

Status GraphRewriteSession::ReplaceNodeWithOutputs(
        GraphNodeId node,
        const std::vector<ReplacementNode>& replacement_nodes) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    for (const ReplacementNode& replacement: replacement_nodes) {
        AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
    }
    const std::array old_nodes{node};
    return ReplaceSubgraph(old_nodes, replacement_nodes);
}

Status GraphRewriteSession::ReplaceSubgraph(
        std::span<const GraphNodeId> old_nodes,
        const std::vector<ReplacementNode>& replacement_nodes) {
    if (old_nodes.empty()) {
        return Status::InvalidArgument("GraphRewriteSession::ReplaceSubgraph old node list is empty");
    }

    for (GraphNodeId old_node: old_nodes) {
        AM_RETURN_IF_ERROR(CheckNodeId(old_node));
    }
    for (const ReplacementNode& replacement: replacement_nodes) {
        AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
    }

    for (GraphNodeId old_node: old_nodes) {
        NodeRewriteEntry& rewrite = node_rewrites_[old_node.index];
        if (rewrite.subgraph_rewrite.has_value()) {
            ClearSubgraphRewrite(*rewrite.subgraph_rewrite);
        }
    }

    SubgraphRewriteEntry subgraph{
            .old_nodes = std::vector<GraphNodeId>(old_nodes.begin(), old_nodes.end()),
            .replacements = replacement_nodes,
    };
    const std::size_t subgraph_index = subgraph_rewrites_.size();
    subgraph_rewrites_.push_back(std::move(subgraph));

    for (GraphNodeId old_node: old_nodes) {
        NodeRewriteEntry& rewrite = node_rewrites_[old_node.index];
        rewrite.kind = NodeRewriteKind::kKeep;
        rewrite.subgraph_rewrite = subgraph_index;
    }
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

        const auto& next = value_replacements_[cur.index];
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
    for (auto& cached_value: resolved_value_cache_) {
        cached_value.reset();
    }
    return Status::Ok();
}

GraphValueId GraphRewriteSession::GetResolvedValue(GraphValueId value) const {
    if (value.index >= value_replacements_.size()) {
        return value;
    }

    if (resolved_value_cache_[value.index].has_value()) {
        return *resolved_value_cache_[value.index];
    }

    std::vector<uint32_t> path;
    GraphValueId cur = value;
    GraphValueId resolved = value;
    for (size_t depth = 0; depth < value_replacements_.size(); ++depth) {
        if (cur.index >= value_replacements_.size()) {
            resolved = cur;
            break;
        }

        if (resolved_value_cache_[cur.index].has_value()) {
            resolved = *resolved_value_cache_[cur.index];
            break;
        }

        path.push_back(cur.index);
        const auto& next = value_replacements_[cur.index];
        if (!next.has_value()) {
            resolved = cur;
            break;
        }
        cur = *next;
        resolved = cur;
    }

    for (uint32_t value_index: path) {
        resolved_value_cache_[value_index] = resolved;
    }
    return resolved;
}

StatusOr<GraphNodeView> GraphRewriteSession::GetNodeView(GraphNodeId node) const {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    if (node_rewrites_[node.index].kind != NodeRewriteKind::kKeep ||
        node_rewrites_[node.index].subgraph_rewrite.has_value()) {
        return Status::NotFound("GraphRewriteSession::GetNodeView node was removed or replaced");
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

    for (const SubgraphRewriteEntry& rewrite: subgraph_rewrites_) {
        for (GraphNodeId old_node: rewrite.old_nodes) {
            AM_RETURN_IF_ERROR(CheckNodeId(old_node));
        }
        for (const ReplacementNode& replacement: rewrite.replacements) {
            AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
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
    for (uint32_t i = 0; i < values.size(); ++i) {
        const GraphValueId old_id{.index = i};
        const GraphValue& value = values[i];
        if (value.producer.has_value()) {
            continue;
        }

        if (std::get_if<ModelInputValue>(&value.payload)) {
            const auto input_name = FindInputName(graph_, old_id);
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
    std::vector<bool> emitted_subgraphs(subgraph_rewrites_.size(), false);
    for (GraphNodeId old_node_id: *order) {
        const NodeRewriteEntry& rewrite = node_rewrites_[old_node_id.index];
        if (rewrite.kind == NodeRewriteKind::kRemove) {
            continue;
        }

        if (rewrite.subgraph_rewrite.has_value()) {
            const std::size_t subgraph_index = *rewrite.subgraph_rewrite;
            if (subgraph_index >= subgraph_rewrites_.size()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::Commit subgraph rewrite index out of range");
            }

            if (!emitted_subgraphs[subgraph_index]) {
                emitted_subgraphs[subgraph_index] = true;
                for (const ReplacementNode& replacement: subgraph_rewrites_[subgraph_index].replacements) {
                    std::vector<GraphValueId> new_inputs;
                    new_inputs.reserve(replacement.inputs.size());
                    for (GraphValueId old_input: replacement.inputs) {
                        const GraphValueId resolved_input = GetResolvedValue(old_input);
                        StatusOr<GraphValueId> mapped_input = MapResolvedValue(resolved_input, value_map);
                        AM_RETURN_IF_ERROR(mapped_input.status());
                        new_inputs.push_back(*mapped_input);
                    }

                    std::vector<NodeOutputDesc> output_descs;
                    output_descs.reserve(replacement.outputs.size());
                    for (const ReplacementOutput& output: replacement.outputs) {
                        output_descs.push_back(output.desc);
                    }

                    const AddedNode added = committed.AddNode(
                            replacement.op_type,
                            replacement.decoder_layer_index,
                            std::move(new_inputs),
                            std::move(output_descs),
                            replacement.op_params,
                            replacement.attrs,
                            replacement.debug_name);

                    for (size_t i = 0; i < replacement.outputs.size(); ++i) {
                        if (replacement.outputs[i].replaces.has_value()) {
                            value_map[replacement.outputs[i].replaces->index] = added.outputs[i];
                        }
                    }
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
            output_descs.push_back(MakeOutputDescFromValue(old_value));
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

Status GraphRewriteSession::ValidateReplacementNode(const ReplacementNode& replacement) const {
    for (GraphValueId input: replacement.inputs) {
        AM_RETURN_IF_ERROR(CheckValueId(input));
    }
    for (const ReplacementOutput& output: replacement.outputs) {
        if (output.replaces.has_value()) {
            AM_RETURN_IF_ERROR(CheckValueId(*output.replaces));
        }
    }
    return Status::Ok();
}

void GraphRewriteSession::ClearSubgraphRewrite(std::size_t subgraph_index) {
    for (NodeRewriteEntry& rewrite: node_rewrites_) {
        if (rewrite.subgraph_rewrite == subgraph_index) {
            rewrite.subgraph_rewrite.reset();
        }
    }
}

const std::vector<GraphValueId>& GraphRewriteSession::CurrentInputs(GraphNodeId node) const {
    if (input_overrides_[node.index].has_value()) {
        return *input_overrides_[node.index];
    }
    return graph_.GetNode(node).inputs;
}

}// namespace aethermind
