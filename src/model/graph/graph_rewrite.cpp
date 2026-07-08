#include "aethermind/model/graph/graph_rewrite.h"
#include "utils/variant_utils.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <variant>

namespace aethermind {
namespace {

// Looks up the model input name associated with a value id. Returns nullopt
// if the value is not a named model input.
std::optional<std::string> FindInputName(const ModelGraph& graph, GraphValueId value) {
    for (const auto& input: graph.GetInputs()) {
        if (input.value == value) {
            return input.name;
        }
    }
    return std::nullopt;
}

// Translates a source value id to its committed counterpart via the value_map.
// Returns InvalidArgument if the value is unmapped (producer removed or not yet emitted).
StatusOr<GraphValueId> MapResolvedValue(GraphValueId old_value,
                                        const std::vector<std::optional<GraphValueId>>& value_map) {
    if (old_value.index >= value_map.size() || !value_map[old_value.index].has_value()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: value " + std::to_string(old_value.index) +
                " cannot be mapped during commit (producer removed or not yet emitted)");
    }
    return *value_map[old_value.index];
}

// Copies all fields from a GraphValue into a NodeOutputDesc for use in a
// replacement node or committed graph node.
NodeOutputDesc MakeOutputDescFromValue(const GraphValue& value) {
    return NodeOutputDesc{
            .spec = value.spec,
            .payload = value.payload,
            .quantization = value.quantization,
            .debug_name = value.debug_name,
    };
}

// Builds a replacement node that is an exact copy of a source graph node,
// with each output binding taking over the original output's value identity.
// Used by RedirectInput to create a mirror rewrite.
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
    // Virtual values occupy the id space starting at graph_.GetValues().size().
    // session_constants_ grows in parallel: nullopt for virtual, SessionConstant for AddConstant.
    const std::size_t next_value_index = graph_.GetValues().size() + virtual_value_count_;
    AM_CHECK(next_value_index < std::numeric_limits<uint32_t>::max(),
             "Graph virtual value id space exhausted");
    ++virtual_value_count_;
    session_constants_.emplace_back(std::nullopt);
    InvalidateConsumerCache();
    return {.index = static_cast<uint32_t>(next_value_index)};
}

GraphValueId GraphRewriteSession::AddConstant(TensorSpec spec,
                                              ConstantBinding binding,
                                              QuantizationSpec quantization,
                                              std::string debug_name) {
    // Same id space as virtual values, but distinguished by a non-nullopt
    // SessionConstant entry in session_constants_.
    const std::size_t next_value_index = graph_.GetValues().size() + virtual_value_count_;
    AM_CHECK(next_value_index < std::numeric_limits<uint32_t>::max(),
             "Graph session constant value id space exhausted");
    ++virtual_value_count_;
    session_constants_.emplace_back(SessionConstant{.spec = std::move(spec),
                                                    .binding = std::move(binding),
                                                    .quantization = quantization,
                                                    .debug_name = std::move(debug_name)});
    InvalidateConsumerCache();
    return {.index = static_cast<uint32_t>(next_value_index)};
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

Status GraphRewriteSession::ReplaceSubgraph(std::span<const GraphNodeId> old_nodes,
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
    InvalidateConsumerCache();
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
    if (existing.has_value()) {
        if (!IsNodeLive(node)) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::RedirectInput cannot redirect a node "
                    "that was removed or replaced by an active rewrite");
        }

        RewriteEntry& rewrite = rewrites_[*existing];
        ReplacementNode& replacement = rewrite.replacements[0];
        // This check defends the mirror-shape invariant before mutating
        // accumulated redirects.
        if (input_index >= replacement.inputs.size()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::RedirectInput replacement input index mismatch");
        }
        replacement.inputs[input_index] = new_value;
        InvalidateConsumerCache();
        return Status::Ok();
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
    InvalidateConsumerCache();
    return Status::Ok();
}

Status GraphRewriteSession::ReplaceValue(GraphValueId old_value, GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckSourceValueId(old_value));
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
    InvalidateConsumerCache();
    return Status::Ok();
}

GraphValueId GraphRewriteSession::GetResolvedValue(GraphValueId value) const {
    if (value.index >= value_replacements_.size()) {
        return value;
    }

    if (resolved_value_cache_[value.index].has_value()) {
        return *resolved_value_cache_[value.index];
    }

    // Walk the replacement chain with path compression: record all visited
    // values in `path`, and after finding the terminal, cache every intermediate
    // to that terminal for O(1) future lookups.
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
    // No rewrite entry -> untouched node, therefore live.
    // Mirror rewrite (RedirectInput) -> exposes original node identity, therefore live.
    // Full subgraph replacement -> replaced/removed, therefore not live.
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
    // Filter the source graph's topological order to include only live nodes.
    // The session does not introduce new edges, so the source ordering is still valid.
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
    if (IsSessionConstant(value)) {
        return true;
    }

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
    if (IsSessionConstant(value)) {
        return MakeOutputDescFromSessionConstant(value);
    }
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
    live.reserve(values.size() + session_constants_.size());
    for (uint32_t i = 0; i < values.size(); ++i) {
        if (const GraphValueId id{.index = i}; IsValueLive(id)) {
            live.push_back(id);
        }
    }
    for (uint32_t i = 0; i < session_constants_.size(); ++i) {
        const GraphValueId id{.index = static_cast<uint32_t>(values.size() + i)};
        if (IsValueLive(id)) {
            live.push_back(id);
        }
    }
    return live;
}

std::vector<GraphNodeId> GraphRewriteSession::FindConsumers(GraphValueId value) const {
    if (value.index >= graph_.GetValues().size() && !IsSessionValue(value)) {
        return {};
    }

    if (IsVirtualValue(value)) {
        return {};
    }

    const GraphValueId resolved_value = GetResolvedValue(value);
    const ConsumerCache& cache = EnsureConsumerCache();
    if (resolved_value.index >= cache.original_consumers.size()) {
        return {};
    }
    return cache.original_consumers[resolved_value.index];
}

bool GraphRewriteSession::HasLiveConsumers(GraphValueId value) const {
    if (value.index >= graph_.GetValues().size() && !IsSessionValue(value)) {
        return false;
    }

    if (IsVirtualValue(value)) {
        return false;
    }

    const GraphValueId resolved_value = GetResolvedValue(value);
    const ConsumerCache& cache = EnsureConsumerCache();
    if (resolved_value.index >= cache.original_consumers.size()) {
        return false;
    }
    return !cache.original_consumers[resolved_value.index].empty() ||
           cache.replacement_consumer_counts[resolved_value.index] > 0;
}

const GraphRewriteSession::ConsumerCache& GraphRewriteSession::EnsureConsumerCache() const {
    if (consumer_cache_.has_value() && consumer_cache_->generation == mutation_generation_) {
        return *consumer_cache_;
    }

    ConsumerCache cache;
    cache.generation = mutation_generation_;
    const std::size_t value_count = graph_.GetValues().size() + virtual_value_count_;
    cache.original_consumers.resize(value_count);
    cache.replacement_consumer_counts.resize(value_count, 0U);

    const StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    AM_CHECK(order.ok(), "EnsureConsumerCache: TopologicalOrder failed: {}",
             order.status().ToString());

    for (const auto node_id: *order) {
        if (!IsNodeLive(node_id)) {
            continue;
        }

        const auto* inputs = &graph_.GetNode(node_id).inputs;
        if (const auto rewrite_index = node_to_rewrite_[node_id.index];
            rewrite_index.has_value()) {
            inputs = &rewrites_[*rewrite_index].replacements[0].inputs;
        }

        std::vector<GraphValueId> consumed_values;
        consumed_values.reserve(inputs->size());
        for (const auto input: *inputs) {
            const auto resolved_input = GetResolvedValue(input);
            if (resolved_input.index >= value_count ||
                std::ranges::find(consumed_values, resolved_input) != consumed_values.end()) {
                continue;
            }
            cache.original_consumers[resolved_input.index].push_back(node_id);
            consumed_values.push_back(resolved_input);
        }
    }

    for (const auto& rewrite: rewrites_) {
        if (!rewrite.active || rewrite.exposes_node_view) {
            continue;
        }

        for (const auto& replacement: rewrite.replacements) {
            for (const auto input: replacement.inputs) {
                const GraphValueId resolved_input = GetResolvedValue(input);
                if (resolved_input.index < value_count) {
                    ++cache.replacement_consumer_counts[resolved_input.index];
                }
            }
        }
    }

    consumer_cache_ = std::move(cache);
    return *consumer_cache_;
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

NodeOutputDesc GraphRewriteSession::MakeOutputDescFromSessionConstant(GraphValueId value) const {
    const SessionConstant& constant = *session_constants_[GetVirtualIndex(value)];
    return NodeOutputDesc{.spec = constant.spec,
                          .payload = ConstantValue{.binding = constant.binding},
                          .quantization = constant.quantization,
                          .debug_name = constant.debug_name};
}

StatusOr<GraphValueId> GraphRewriteSession::MapCommittedValue(
        GraphValueId value,
        const CommitValueMaps& maps) const {
    if (value.index < graph_.GetValues().size()) {
        return MapResolvedValue(value, maps.source_values);
    }

    const std::size_t session_index = GetVirtualIndex(value);
    if (session_index >= virtual_value_count_) {
        return Status::InvalidArgument(
                "GraphRewriteSession: session value id out of range during commit");
    }

    if (IsSessionConstant(value)) {
        if (session_index >= maps.session_constants.size() || !maps.session_constants[session_index].has_value()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession: session constant cannot be mapped during commit");
        }
        return *maps.session_constants[session_index];
    }

    if (session_index >= maps.virtual_values.size() || !maps.virtual_values[session_index].has_value()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: virtual value " + std::to_string(value.index) +
                " cannot be mapped during commit (not produced within its rewrite)");
    }
    return *maps.virtual_values[session_index];
}

Status GraphRewriteSession::CopyExternalValues(ModelGraph& committed,
                                               CommitValueMaps& maps) const {
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
            maps.source_values[i] = committed.AddInput(value.spec, *input_name);
        } else if (const auto* weight = std::get_if<WeightValue>(&value.payload)) {
            maps.source_values[i] = committed.AddWeight(value.spec, weight->binding, value.debug_name);
        } else if (const auto* constant = std::get_if<ConstantValue>(&value.payload)) {
            maps.source_values[i] = committed.AddConstant(value.spec, constant->binding, value.debug_name);
        } else if (const auto* state = std::get_if<StateValue>(&value.payload)) {
            maps.source_values[i] = committed.AddState(value.spec, state->binding, value.debug_name);
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

        committed.SetQuantization(*maps.source_values[i], value.quantization);
    }

    for (uint32_t i = 0; i < session_constants_.size(); ++i) {
        if (!session_constants_[i].has_value()) {
            continue;
        }

        const SessionConstant& constant = *session_constants_[i];
        maps.session_constants[i] = committed.AddConstant(constant.spec,
                                                          constant.binding,
                                                          constant.debug_name);
        committed.SetQuantization(*maps.session_constants[i], constant.quantization);
    }
    return Status::Ok();
}

Status GraphRewriteSession::EmitRewrite(const RewriteEntry& rewrite,
                                        ModelGraph& committed,
                                        CommitValueMaps& maps) const {
    // For each replacement node, resolve and map inputs, add the node to
    // the committed graph, then map each output through the replaces binding
    // into the appropriate map (source_values, virtual_values, or error).
    for (const ReplacementNode& replacement: rewrite.replacements) {
        std::vector<GraphValueId> new_inputs;
        new_inputs.reserve(replacement.inputs.size());
        for (GraphValueId input: replacement.inputs) {
            const GraphValueId resolved_input = GetResolvedValue(input);
            StatusOr<GraphValueId> mapped_input = MapCommittedValue(resolved_input, maps);
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
                    if (maps.virtual_values[GetVirtualIndex(replaced)].has_value()) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession::Commit replacement virtual value was already mapped");
                    }
                    maps.virtual_values[GetVirtualIndex(replaced)] = added.outputs[i];
                } else if (IsSessionConstant(replaced)) {
                    return Status::InvalidArgument(
                            "GraphRewriteSession::Commit replacement cannot produce a session constant");
                } else {
                    if (maps.source_values[replaced.index].has_value()) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession::Commit replacement value was already mapped");
                    }
                    maps.source_values[replaced.index] = added.outputs[i];
                }
            }
        }
    }
    return Status::Ok();
}

Status GraphRewriteSession::EmitOriginalNode(GraphNodeId old_node,
                                             ModelGraph& committed,
                                             CommitValueMaps& maps) const {
    // Emit a surviving original node (untouched or RedirectInput'd) into the
    // committed graph. Uses GetNodeView to get the resolved input view and
    // faithfully reproduces the node's outputs.
    StatusOr<GraphNodeView> view = GetNodeView(old_node);
    AM_RETURN_IF_ERROR(view.status());

    std::vector<GraphValueId> new_inputs;
    new_inputs.reserve(view->inputs.size());
    for (GraphValueId input: view->inputs) {
        const GraphValueId resolved_input = GetResolvedValue(input);
        StatusOr<GraphValueId> mapped_input = MapCommittedValue(resolved_input, maps);
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
        if (maps.source_values[view->outputs[i].index].has_value()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::Commit original node output was already mapped");
        }
        maps.source_values[view->outputs[i].index] = added.outputs[i];
    }
    return Status::Ok();
}

Status GraphRewriteSession::MarkCommittedOutputs(ModelGraph& committed,
                                                 const CommitValueMaps& maps) const {
    // Graph outputs are resolved through ReplaceValue chains before mapping
    // into the committed graph's value space.
    for (const auto& output: graph_.GetOutputs()) {
        const GraphValueId resolved_output = GetResolvedValue(output.value);
        StatusOr<GraphValueId> mapped_output = MapCommittedValue(resolved_output, maps);
        AM_RETURN_IF_ERROR(mapped_output.status());
        committed.MarkOutput(*mapped_output, output.name);
    }
    return Status::Ok();
}

StatusOr<ModelGraph> GraphRewriteSession::Commit() const {
    AM_RETURN_IF_ERROR(ValidateEdits());

    ModelGraph committed(graph_.GetConfig());
    ValueMap value_map(graph_.GetValues().size(), std::nullopt);
    ValueMap session_constant_map(virtual_value_count_, std::nullopt);
    ValueMap virtual_value_map(virtual_value_count_, std::nullopt);
    CommitValueMaps maps{.source_values = value_map,
                         .session_constants = session_constant_map,
                         .virtual_values = virtual_value_map};

    AM_RETURN_IF_ERROR(CopyExternalValues(committed, maps));

    // Emit nodes in source topological order. When a node is the first in
    // topological order among its rewrite's old_nodes, emit the entire rewrite
    // (all replacement nodes) before continuing. Live, untouched nodes are
    // emitted as-is via EmitOriginalNode.
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
                AM_RETURN_IF_ERROR(EmitRewrite(rewrite, committed, maps));
            }
            continue;
        }

        AM_RETURN_IF_ERROR(EmitOriginalNode(old_node_id, committed, maps));
    }

    AM_RETURN_IF_ERROR(MarkCommittedOutputs(committed, maps));
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
    if (value.index < graph_.GetValues().size() || IsSessionConstant(value)) {
        return Status::Ok();
    }
    return Status::InvalidArgument("GraphRewriteSession: value id out of range");
}

Status GraphRewriteSession::CheckSourceValueId(GraphValueId value) const {
    if (value.index >= graph_.GetValues().size()) {
        return Status::InvalidArgument("GraphRewriteSession: value id out of range");
    }
    return Status::Ok();
}

Status GraphRewriteSession::CheckValueIdAllowVirtual(GraphValueId value) const {
    // IsSessionValue() already bounds-checks the virtual index against
    // virtual_value_count_, so any session value reaching the second clause
    // is a valid virtual value — no extra range check needed.
    if (value.index < graph_.GetValues().size() || IsSessionConstant(value) ||
        IsSessionValue(value)) {
        return Status::Ok();
    }
    return Status::InvalidArgument("GraphRewriteSession: value id out of range");
}

bool GraphRewriteSession::IsSessionValue(GraphValueId value) const noexcept {
    return value.index >= graph_.GetValues().size() &&
           GetVirtualIndex(value) < virtual_value_count_;
}

bool GraphRewriteSession::IsSessionConstant(GraphValueId value) const noexcept {
    if (!IsSessionValue(value)) {
        return false;
    }

    const std::size_t session_index = GetVirtualIndex(value);
    return session_index < session_constants_.size() &&
           session_constants_[session_index].has_value();
}

bool GraphRewriteSession::IsVirtualValue(GraphValueId value) const noexcept {
    return IsSessionValue(value) && !IsSessionConstant(value);
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
    // Two-level tracking: globally_produced prevents duplicate production of
    // any virtual value across all rewrites. locally_available (per rewrite)
    // ensures each virtual value is produced before it is consumed within the
    // same rewrite group.
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
    // Idempotent: no-op if the rewrite is already inactive or out of range.
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

void GraphRewriteSession::InvalidateConsumerCache() noexcept {
    ++mutation_generation_;
    consumer_cache_.reset();
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
    // Each output descriptor gets a freshly allocated virtual value; these
    // virtual values are bound via RewriteOutputBinding::replaces and can
    // be consumed by subsequent Emit calls or redirected by Yield.
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
    AM_RETURN_IF_ERROR(session_.CheckSourceValueId(old_value_to_replace));

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
    // Submit accumulated replacement nodes; on success, clear internal state
    // so the builder can be reused for another Emit/Yield/Commit cycle.
    Status status = session_.ReplaceSubgraph(old_nodes_, new_nodes_);
    if (status.ok()) {
        new_nodes_.clear();
    }
    return status;
}

}// namespace aethermind
