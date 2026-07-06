#include "aethermind/model/graph/graph_rewrite.h"
#include "utils/variant_utils.h"

#include <algorithm>
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
                "GraphRewriteSession: value " + std::to_string(old_value.index) +
                " cannot be mapped during commit (producer removed or not yet emitted)");
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
                "GraphRewriteSession: virtual value " + std::to_string(value.index) +
                " cannot be mapped during commit (not produced within its rewrite)");
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
    AM_RETURN_IF_ERROR(ValidateReplacementTargets(old_nodes, replacement_nodes));

    for (GraphNodeId old_node: old_nodes) {
        if (const auto rewrite_index = node_to_rewrite_[old_node.index];
            rewrite_index.has_value()) {
            DeactivateRewrite(*rewrite_index);
        }
    }

    const std::size_t rewrite_index = rewrites_.size();
    rewrites_.push_back({
            .old_nodes = {old_nodes.begin(), old_nodes.end()},
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

    // IsNodeLive returns true both for untouched nodes and for nodes with a
    // live single-node mirror rewrite. Only the latter can be mutated in place,
    // so guard with has_value() to exclude the untouched-node case.
    const auto existing = node_to_rewrite_[node.index];
    if (existing.has_value() && IsNodeLive(node)) {
        RewriteEntry& rewrite = rewrites_[*existing];
        ReplacementNode& replacement = rewrite.replacements[0];
        // This check defends the mirror-shape invariant before mutating
        // accumulated redirects.
        if (input_index >= replacement.inputs.size()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::RedirectInput replacement input index mismatch");
        }
        replacement.inputs[input_index] = new_value;
        return Status::Ok();
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

    for (auto value_index: path) {
        resolved_value_cache_[value_index] = resolved;
    }
    return resolved;
}

StatusOr<GraphNodeView> GraphRewriteSession::GetNodeView(GraphNodeId node) const {
    AM_RETURN_IF_ERROR(CheckNodeId(node));

    const GraphNode& original = graph_.GetNode(node);
    const ReplacementNode* replacement = nullptr;

    if (const auto rewrite_opt = node_to_rewrite_[node.index]; rewrite_opt.has_value()) {
        if (!IsNodeLive(node)) {
            return Status::NotFound(
                    "GraphRewriteSession::GetNodeView node was removed or replaced");
        }
        replacement = &rewrites_[*rewrite_opt].replacements[0];
    }

    // Outputs always come from the original node: a RedirectInput mirror
    // replacement only rewrites input wiring, not the produced values.
    // The remaining fields come from the mirror replacement when present,
    // otherwise from the original node.
    GraphNodeView view{
            .node = node,
            .op_type = replacement ? replacement->op_type : original.op_type,
            .decoder_layer_index = replacement ? replacement->decoder_layer_index
                                               : original.decoder_layer_index,
            .inputs = replacement ? replacement->inputs : original.inputs,
            .outputs = original.outputs,
            .attrs = replacement ? replacement->attrs : original.attrs,
            .op_params = replacement ? replacement->op_params : original.op_params,
            .debug_name = replacement ? replacement->debug_name : original.debug_name,
    };

    for (GraphValueId& input: view.inputs) {
        input = GetResolvedValue(input);
    }
    return view;
}

bool GraphRewriteSession::IsNodeLive(GraphNodeId node) const noexcept {
    if (node.index >= node_to_rewrite_.size()) {
        return false;
    }

    const auto& rewrite_opt = node_to_rewrite_[node.index];
    if (!rewrite_opt.has_value()) {
        return true;
    }

    const RewriteEntry& rewrite = rewrites_[*rewrite_opt];
    return rewrite.active && rewrite.exposes_node_view &&
           rewrite.old_nodes.size() == 1 && rewrite.old_nodes[0] == node &&
           rewrite.replacements.size() == 1;
}

StatusOr<std::vector<GraphNodeId>> GraphRewriteSession::GetTopologicalOrder() const {
    StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());

    std::vector<GraphNodeId> live;
    live.reserve(order->size());
    for (GraphNodeId id: *order) {
        if (IsNodeLive(id)) {
            live.push_back(id);
        }
    }
    return live;
}

std::vector<GraphNodeId> GraphRewriteSession::FindNodesByOpType(OpType op_type) const {
    const std::vector<GraphNodeId> candidates = graph_.FindNodesByOpType(op_type);
    std::vector<GraphNodeId> live;
    live.reserve(candidates.size());
    for (GraphNodeId id: candidates) {
        if (IsNodeLive(id)) {
            live.push_back(id);
        }
    }
    return live;
}

bool GraphRewriteSession::IsValueLive(GraphValueId value) const noexcept {
    if (value.index >= graph_.GetValues().size()) {
        return false;
    }

    const std::optional<GraphNodeId> producer = graph_.GetValue(value).producer;
    if (!producer.has_value()) {
        return true;
    }

    if (IsNodeLive(*producer)) {
        return true;
    }

    // Producer removed/replaced, but an active rewrite may take over this value
    // via a replacement output's `replaces` binding.
    return IsValueReplacedByActiveRewrite(value);
}

StatusOr<NodeOutputDesc> GraphRewriteSession::GetValueOutputDesc(GraphValueId value) const {
    AM_RETURN_IF_ERROR(CheckValueId(value));
    return MakeOutputDescFromValue(graph_.GetValue(value));
}

bool GraphRewriteSession::IsGraphOutput(GraphValueId value) const noexcept {
    if (value.index >= graph_.GetValues().size()) {
        return false;
    }

    return std::ranges::any_of(graph_.GetOutputs(), [&](const auto& output) {
        return output.value == value;
    });
}

bool GraphRewriteSession::IsValueReplacedByActiveRewrite(GraphValueId value) const noexcept {
    for (const RewriteEntry& rewrite: rewrites_) {
        if (!rewrite.active) {
            continue;
        }

        for (const ReplacementNode& replacement: rewrite.replacements) {
            for (const RewriteOutputBinding& output: replacement.outputs) {
                if (output.replaces == value) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::vector<GraphValueId> GraphRewriteSession::GetLiveValues() const {
    const std::span<const GraphValue> values = graph_.GetValues();
    std::vector<GraphValueId> live;
    live.reserve(values.size());
    for (uint32_t i = 0; i < values.size(); ++i) {
        if (const GraphValueId id{.index = i}; IsValueLive(id)) {
            live.push_back(id);
        }
    }
    return live;
}

std::vector<GraphNodeId> GraphRewriteSession::FindConsumers(GraphValueId value) const {
    const GraphValueId resolved_value = GetResolvedValue(value);

    const StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    if (!order.ok()) {
        return {};
    }

    std::vector<GraphNodeId> consumers;
    for (const GraphNodeId node_id: *order) {
        if (!IsNodeLive(node_id)) {
            continue;
        }
        const StatusOr<GraphNodeView> view = GetNodeView(node_id);
        if (!view.ok()) {
            continue;
        }
        // GetNodeView already resolves inputs via GetResolvedValue, so direct
        // comparison with resolved_value is correct.
        for (const GraphValueId input: view->inputs) {
            if (input == resolved_value) {
                consumers.push_back(node_id);
                break;
            }
        }
    }
    return consumers;
}

Status GraphRewriteSession::ValidateEdits() const {
    for (const auto& replacement: value_replacements_) {
        if (replacement.has_value()) {
            AM_RETURN_IF_ERROR(CheckValueId(*replacement));
        }
    }

    for (const auto& rewrite: rewrites_) {
        if (!rewrite.active) {
            continue;
        }

        for (auto old_node: rewrite.old_nodes) {
            AM_RETURN_IF_ERROR(CheckNodeId(old_node));
        }

        for (const auto& replacement: rewrite.replacements) {
            AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
        }
        AM_RETURN_IF_ERROR(ValidateReplacementTargets(rewrite.old_nodes, rewrite.replacements));
    }
    AM_RETURN_IF_ERROR(ValidateVirtualValues());
    return Status::Ok();
}

Status GraphRewriteSession::CopyExternalValues(ModelGraph& committed, ValueMap& value_map) const {
    const std::span<const GraphValue> values = graph_.GetValues();
    for (uint32_t i = 0; i < values.size(); ++i) {
        const GraphValue& value = values[i];
        if (value.producer.has_value()) {
            continue;
        }

        if (std::get_if<ModelInputValue>(&value.payload)) {
            const auto input_name = FindInputName(graph_, {.index = i});
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
    return Status::Ok();
}

Status GraphRewriteSession::EmitRewrite(const RewriteEntry& rewrite,
                                        ModelGraph& committed,
                                        ValueMap& value_map,
                                        ValueMap& virtual_value_map) const {
    for (const ReplacementNode& replacement: rewrite.replacements) {
        std::vector<GraphValueId> new_inputs;
        new_inputs.reserve(replacement.inputs.size());
        for (GraphValueId input: replacement.inputs) {
            const GraphValueId resolved_input = GetResolvedValue(input);
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
                    if (virtual_value_map[GetVirtualIndex(replaced)].has_value()) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession::Commit replacement virtual value was already mapped");
                    }
                    virtual_value_map[GetVirtualIndex(replaced)] = added.outputs[i];
                } else {
                    if (value_map[replaced.index].has_value()) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession::Commit replacement value was already mapped");
                    }
                    value_map[replaced.index] = added.outputs[i];
                }
            }
        }
    }
    return Status::Ok();
}

Status GraphRewriteSession::EmitOriginalNode(GraphNodeId old_node,
                                             ModelGraph& committed,
                                             ValueMap& value_map) const {
    StatusOr<GraphNodeView> view = GetNodeView(old_node);
    AM_RETURN_IF_ERROR(view.status());

    std::vector<GraphValueId> new_inputs;
    new_inputs.reserve(view->inputs.size());
    for (GraphValueId input: view->inputs) {
        const GraphValueId resolved_input = GetResolvedValue(input);
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
        if (value_map[view->outputs[i].index].has_value()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::Commit original node output was already mapped");
        }
        value_map[view->outputs[i].index] = added.outputs[i];
    }
    return Status::Ok();
}

Status GraphRewriteSession::MarkCommittedOutputs(ModelGraph& committed, const ValueMap& value_map) const {
    for (const auto& output: graph_.GetOutputs()) {
        const GraphValueId resolved_output = GetResolvedValue(output.value);
        StatusOr<GraphValueId> mapped_output = MapResolvedValue(resolved_output, value_map);
        AM_RETURN_IF_ERROR(mapped_output.status());
        committed.MarkOutput(*mapped_output, output.name);
    }
    return Status::Ok();
}

StatusOr<ModelGraph> GraphRewriteSession::Commit() const {
    AM_RETURN_IF_ERROR(ValidateEdits());

    ModelGraph committed(graph_.GetConfig());
    ValueMap value_map(graph_.GetValues().size(), std::nullopt);
    ValueMap virtual_value_map(virtual_value_count_, std::nullopt);

    AM_RETURN_IF_ERROR(CopyExternalValues(committed, value_map));

    StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());
    std::vector emitted_rewrites(rewrites_.size(), false);
    for (GraphNodeId old_node_id: *order) {
        if (const auto rewrite_index = node_to_rewrite_[old_node_id.index];
            rewrite_index.has_value()) {
            if (*rewrite_index >= rewrites_.size()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::Commit rewrite index out of range");
            }

            const auto& rewrite = rewrites_[*rewrite_index];
            if (!rewrite.active) {
                return Status::InvalidArgument(
                        "GraphRewriteSession::Commit inactive rewrite is still referenced");
            }

            if (!emitted_rewrites[*rewrite_index]) {
                emitted_rewrites[*rewrite_index] = true;
                AM_RETURN_IF_ERROR(EmitRewrite(rewrite, committed, value_map, virtual_value_map));
            }
            continue;
        }

        AM_RETURN_IF_ERROR(EmitOriginalNode(old_node_id, committed, value_map));
    }

    AM_RETURN_IF_ERROR(MarkCommittedOutputs(committed, value_map));
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

Status GraphRewriteSession::ValidateReplacementTargets(
        std::span<const GraphNodeId> old_nodes,
        const std::vector<ReplacementNode>& replacement_nodes) const {
    std::vector<GraphValueId> replaceable_outputs;
    for (GraphNodeId old_node: old_nodes) {
        AM_RETURN_IF_ERROR(CheckNodeId(old_node));
        const GraphNode& node = graph_.GetNode(old_node);
        replaceable_outputs.insert(replaceable_outputs.end(),
                                   node.outputs.begin(), node.outputs.end());
    }

    std::vector<GraphValueId> real_replacements;
    for (const ReplacementNode& replacement: replacement_nodes) {
        for (const RewriteOutputBinding& output: replacement.outputs) {
            if (!output.replaces.has_value() || IsVirtualValue(*output.replaces)) {
                continue;
            }

            const GraphValueId replaced = *output.replaces;
            if (std::ranges::find(replaceable_outputs, replaced) == replaceable_outputs.end()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession: replacement output target is "
                        "not produced by replaced old_nodes");
            }

            if (std::ranges::find(real_replacements, replaced) != real_replacements.end()) {
                return Status::InvalidArgument(
                        "GraphRewriteSession: replacement output target is "
                        "produced more than once");
            }
            real_replacements.push_back(replaced);
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
                    if (!locally_available[GetVirtualIndex(input)]) {
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

GraphValueId SubgraphBuilder::Emit(OpType op_type,
                                   std::vector<GraphValueId> inputs,
                                   NodeOutputDesc output_desc,
                                   OpParams op_params,
                                   std::optional<uint32_t> decoder_layer_index,
                                   std::string debug_name) {
    std::vector<NodeOutputDesc> output_descs;
    output_descs.push_back(std::move(output_desc));
    std::vector<GraphValueId> outputs = Emit(op_type,
                                             std::move(inputs),
                                             std::move(output_descs),
                                             std::move(op_params),
                                             decoder_layer_index,
                                             std::move(debug_name));
    AM_CHECK(outputs.size() == 1, "SubgraphBuilder::Emit single-output wrapper expected one output");
    return outputs[0];
}

std::vector<GraphValueId> SubgraphBuilder::Emit(OpType op_type,
                                                std::vector<GraphValueId> inputs,
                                                std::vector<NodeOutputDesc> output_descs,
                                                OpParams op_params,
                                                std::optional<uint32_t> decoder_layer_index,
                                                std::string debug_name) {
    std::vector<GraphValueId> virtual_ids;
    virtual_ids.reserve(output_descs.size());

    ReplacementNode node{
            .op_type = op_type,
            .decoder_layer_index = decoder_layer_index,
            .inputs = std::move(inputs),
            .op_params = std::move(op_params),
            .debug_name = std::move(debug_name),
    };

    node.outputs.reserve(output_descs.size());
    for (NodeOutputDesc& output_desc: output_descs) {
        const GraphValueId virtual_id = session_.AllocateVirtualValue();
        virtual_ids.push_back(virtual_id);
        node.outputs.push_back(RewriteOutputBinding{
                .desc = std::move(output_desc),
                .replaces = virtual_id,
        });
    }

    new_nodes_.push_back(std::move(node));
    return virtual_ids;
}

Status SubgraphBuilder::Yield(GraphValueId internal_val, GraphValueId old_value_to_replace) {
    // Yield redirects an internal virtual value to replace an external real
    // graph value. Reject virtual or out-of-range ids for old_value_to_replace
    // early, so the error is attributed to the caller rather than surfacing
    // later as a confusing ValidateVirtualValues or Commit failure.
    AM_RETURN_IF_ERROR(session_.CheckValueId(old_value_to_replace));

    for (auto& node: new_nodes_) {
        for (auto& out: node.outputs) {
            if (out.replaces == internal_val) {
                out.replaces = old_value_to_replace;
                return Status::Ok();
            }
        }
    }
    return Status::InvalidArgument(
            "SubgraphBuilder::Yield: internal_val was not produced by any Emit call");
}

Status SubgraphBuilder::Commit() {
    Status status = session_.ReplaceSubgraph(old_nodes_, new_nodes_);
    if (status.ok()) {
        new_nodes_.clear();
    }
    return status;
}

}// namespace aethermind
