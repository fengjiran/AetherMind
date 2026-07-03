#include "aethermind/model/graph/graph_rewrite.h"
#include "utils/variant_utils.h"

#include <array>
#include <limits>
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

StatusOr<GraphValueId> MapCommittedValue(
        GraphValueId value,
        std::size_t real_value_count,
        const std::vector<std::optional<GraphValueId>>& value_map,
        const std::vector<std::optional<GraphValueId>>& virtual_value_map) {
    if (value.index < real_value_count) {
        return MapResolvedValue(value, value_map);
    }

    const std::size_t virtual_index = value.index - real_value_count;
    if (virtual_index >= virtual_value_map.size() || !virtual_value_map[virtual_index].has_value()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: virtual value cannot be mapped during commit");
    }
    return *virtual_value_map[virtual_index];
}

NodeOutputDesc MakeOutputDescFromValue(const GraphValue& value) {
    return NodeOutputDesc{
            .spec = value.spec,
            .payload = value.payload,
            .quantization = value.quantization,
            .debug_name = value.debug_name,
    };
}

ReplacementNode BuildMirrorReplacement(const ModelGraph& graph, GraphNodeId node) {
    const GraphNode& original = graph.GetNode(node);
    ReplacementNode rn{
            .op_type = original.op_type,
            .decoder_layer_index = original.decoder_layer_index,
            .inputs = original.inputs,
            .attrs = original.attrs,
            .op_params = original.op_params,
            .debug_name = original.debug_name,
    };

    rn.outputs.reserve(original.outputs.size());
    for (GraphValueId output: original.outputs) {
        const GraphValue& value = graph.GetValue(output);
        rn.outputs.push_back(RewriteOutputBinding{
                .desc = MakeOutputDescFromValue(value),
                .replaces = output,
        });
    }
    return rn;
}

}// namespace

GraphRewriteSession::GraphRewriteSession(const ModelGraph& graph)
    : graph_(graph),
      node_to_rewrite_(graph.GetNodes().size(), std::nullopt),
      value_replacements_(graph.GetValues().size(), std::nullopt),
      resolved_value_cache_(graph.GetValues().size(), std::nullopt) {}

GraphValueId GraphRewriteSession::AllocateVirtualValue() {
    const std::size_t next_value_index = graph_.GetValues().size() + virtual_value_count_;
    AM_CHECK(next_value_index < std::numeric_limits<uint32_t>::max(),
             "Graph virtual value id space exhausted");
    ++virtual_value_count_;
    return GraphValueId{.index = static_cast<uint32_t>(next_value_index)};
}

Status GraphRewriteSession::Apply(std::span<const GraphMutation> mutations) {
    auto visitor = overloaded{
            [this](const SubgraphReplacement& replace) {
                return ReplaceSubgraph(replace.old_nodes, replace.replacement_nodes);
            },
            [this](const NodeRemoval& remove) {
                return RemoveNode(remove.node);
            },
            [this](const InputRedirection& redirect) {
                return RedirectInput(redirect.node, redirect.input_index, redirect.new_value);
            },
            [this](const ValueReplacement& replace) {
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
    const std::array old_nodes{node};
    return ReplaceSubgraph(old_nodes, {});
}

Status GraphRewriteSession::ReplaceSubgraph(
        std::span<const GraphNodeId> old_nodes,
        const std::vector<ReplacementNode>& replacement_nodes) {
    if (old_nodes.empty()) {
        return Status::InvalidArgument(
                "GraphRewriteSession::ReplaceSubgraph old node list is empty");
    }

    for (GraphNodeId old_node: old_nodes) {
        AM_RETURN_IF_ERROR(CheckNodeId(old_node));
    }

    for (const auto& replacement: replacement_nodes) {
        AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
    }

    for (GraphNodeId old_node: old_nodes) {
        if (const auto rewrite_index = node_to_rewrite_[old_node.index];
            rewrite_index.has_value()) {
            DeactivateRewrite(*rewrite_index);
        }
    }

    const std::size_t rewrite_index = rewrites_.size();
    rewrites_.push_back({
            .old_nodes = std::vector<GraphNodeId>(old_nodes.begin(), old_nodes.end()),
            .replacements = replacement_nodes,
    });

    for (GraphNodeId old_node: old_nodes) {
        node_to_rewrite_[old_node.index] = rewrite_index;
    }
    return Status::Ok();
}

Status GraphRewriteSession::RedirectInput(GraphNodeId node, size_t input_index,
                                          GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckNodeId(node));
    AM_RETURN_IF_ERROR(CheckValueId(new_value));
    if (input_index >= graph_.GetNode(node).inputs.size()) {
        return Status::InvalidArgument(
                "GraphRewriteSession::RedirectInput input index out of range");
    }

    const auto existing = node_to_rewrite_[node.index];
    if (existing.has_value() &&
        rewrites_[*existing].active &&
        rewrites_[*existing].exposes_node_view) {
        if (RewriteEntry& rewrite = rewrites_[*existing];
            rewrite.old_nodes.size() == 1 && rewrite.old_nodes[0] == node &&
            rewrite.replacements.size() == 1) {
            ReplacementNode& replacement = rewrite.replacements[0];
            // RedirectInput-created rewrites mirror the original node shape;
            // this check defends that invariant before mutating accumulated redirects.
            if (input_index >= replacement.inputs.size()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::RedirectInput replacement input index mismatch");
            }
            replacement.inputs[input_index] = new_value;
            return Status::Ok();
        }
    }

    if (existing.has_value()) {
        DeactivateRewrite(*existing);
    }

    auto replacement = BuildMirrorReplacement(graph_, node);
    replacement.inputs[input_index] = new_value;

    const std::size_t idx = rewrites_.size();
    rewrites_.push_back({
            .old_nodes = {node},
            .replacements = {std::move(replacement)},
            .exposes_node_view = true,
    });
    node_to_rewrite_[node.index] = idx;
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

    if (const auto rewrite_opt = node_to_rewrite_[node.index]; rewrite_opt.has_value()) {
        const RewriteEntry& rewrite = rewrites_[*rewrite_opt];
        if (!rewrite.active || !rewrite.exposes_node_view ||
            rewrite.old_nodes.size() != 1 || rewrite.old_nodes[0] != node ||
            rewrite.replacements.size() != 1) {
            return Status::NotFound(
                    "GraphRewriteSession::GetNodeView node was removed or replaced");
        }
        const GraphNode& original = graph_.GetNode(node);
        const ReplacementNode& replacement = rewrite.replacements[0];

        GraphNodeView view{
                .node = node,
                .op_type = replacement.op_type,
                .decoder_layer_index = replacement.decoder_layer_index,
                .inputs = replacement.inputs,
                .outputs = original.outputs,
                .attrs = replacement.attrs,
                .op_params = replacement.op_params,
                .debug_name = replacement.debug_name,
        };
        for (GraphValueId& input: view.inputs) {
            input = GetResolvedValue(input);
        }
        return view;
    }

    const GraphNode& original = graph_.GetNode(node);
    GraphNodeView view{
            .node = node,
            .op_type = original.op_type,
            .decoder_layer_index = original.decoder_layer_index,
            .inputs = original.inputs,
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
    for (const auto& replacement: value_replacements_) {
        if (replacement.has_value()) {
            AM_RETURN_IF_ERROR(CheckValueId(*replacement));
        }
    }

    for (const auto& rewrite: rewrites_) {
        for (auto old_node: rewrite.old_nodes) {
            AM_RETURN_IF_ERROR(CheckNodeId(old_node));
        }

        for (const auto& replacement: rewrite.replacements) {
            AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
        }
    }
    AM_RETURN_IF_ERROR(ValidateVirtualValues());
    return Status::Ok();
}

StatusOr<ModelGraph> GraphRewriteSession::Commit() const {
    AM_RETURN_IF_ERROR(ValidateEdits());

    ModelGraph committed(graph_.GetConfig());
    std::vector<std::optional<GraphValueId>> value_map(
            graph_.GetValues().size(), std::nullopt);
    std::vector<std::optional<GraphValueId>> virtual_value_map(
            virtual_value_count_, std::nullopt);

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
    std::vector<bool> emitted_rewrites(rewrites_.size(), false);
    for (GraphNodeId old_node_id: *order) {
        const auto rewrite_index = node_to_rewrite_[old_node_id.index];
        if (rewrite_index.has_value()) {
            if (*rewrite_index >= rewrites_.size()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::Commit rewrite index out of range");
            }
            const RewriteEntry& rewrite = rewrites_[*rewrite_index];
            if (!rewrite.active) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::Commit inactive rewrite is still referenced");
            }

            if (!emitted_rewrites[*rewrite_index]) {
                emitted_rewrites[*rewrite_index] = true;
                for (const ReplacementNode& replacement: rewrite.replacements) {
                    std::vector<GraphValueId> new_inputs;
                    new_inputs.reserve(replacement.inputs.size());
                    for (GraphValueId old_input: replacement.inputs) {
                        const GraphValueId resolved_input = GetResolvedValue(old_input);
                        StatusOr<GraphValueId> mapped_input = MapCommittedValue(
                                resolved_input, graph_.GetValues().size(), value_map, virtual_value_map);
                        AM_RETURN_IF_ERROR(mapped_input.status());
                        new_inputs.push_back(*mapped_input);
                    }

                    std::vector<NodeOutputDesc> output_descs;
                    output_descs.reserve(replacement.outputs.size());
                    for (const RewriteOutputBinding& output: replacement.outputs) {
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
                            const GraphValueId replaced = *replacement.outputs[i].replaces;
                            if (IsVirtualValue(replaced)) {
                                virtual_value_map[GetVirtualIndex(replaced)] = added.outputs[i];
                            } else {
                                value_map[replaced.index] = added.outputs[i];
                            }
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

Status GraphRewriteSession::CheckValueIdAllowVirtual(GraphValueId value) const {
    if (IsVirtualValue(value)) {
        if (GetVirtualIndex(value) < virtual_value_count_) {
            return Status::Ok();
        }
        return Status::InvalidArgument(
                "GraphRewriteSession: virtual value id out of range");
    }
    return CheckValueId(value);
}

bool GraphRewriteSession::IsVirtualValue(GraphValueId value) const noexcept {
    return value.index >= graph_.GetValues().size();
}

Status GraphRewriteSession::ValidateReplacementNode(const ReplacementNode& replacement) const {
    for (GraphValueId input: replacement.inputs) {
        AM_RETURN_IF_ERROR(CheckValueIdAllowVirtual(input));
    }

    for (const auto& output: replacement.outputs) {
        if (output.replaces.has_value()) {
            AM_RETURN_IF_ERROR(CheckValueIdAllowVirtual(*output.replaces));
        }
    }
    return Status::Ok();
}

Status GraphRewriteSession::ValidateVirtualValues() const {
    std::vector globally_produced(virtual_value_count_, false);

    for (const auto& rewrite: rewrites_) {
        if (!rewrite.active) {
            continue;
        }

        std::vector locally_available(virtual_value_count_, false);
        for (const auto& replacement: rewrite.replacements) {
            for (GraphValueId input: replacement.inputs) {
                if (IsVirtualValue(input)) {
                    if (const std::size_t virtual_index = GetVirtualIndex(input);
                        !locally_available[virtual_index]) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession: virtual value is consumed before being produced");
                    }
                }
            }

            for (const auto& output: replacement.outputs) {
                if (!output.replaces.has_value() || !IsVirtualValue(*output.replaces)) {
                    continue;
                }

                const std::size_t virtual_index = GetVirtualIndex(*output.replaces);
                if (globally_produced[virtual_index]) {
                    return Status::InvalidArgument(
                            "GraphRewriteSession: virtual value produced more than once");
                }
                globally_produced[virtual_index] = true;
                locally_available[virtual_index] = true;
            }
        }
    }
    return Status::Ok();
}

void GraphRewriteSession::DeactivateRewrite(std::size_t rewrite_index) {
    if (rewrite_index >= rewrites_.size() || !rewrites_[rewrite_index].active) {
        return;
    }

    rewrites_[rewrite_index].active = false;
    for (GraphNodeId old_node: rewrites_[rewrite_index].old_nodes) {
        if (old_node.index < node_to_rewrite_.size() &&
            node_to_rewrite_[old_node.index] == rewrite_index) {
            node_to_rewrite_[old_node.index].reset();
        }
    }
}

}// namespace aethermind
