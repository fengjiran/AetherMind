#include "aethermind/graph/optimization/graph_rewrite.h"
#include "aethermind/operators/operator_inference.h"
#include "utils/variant_utils.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <variant>

namespace aethermind {
namespace {

// Looks up the model input name associated with a value id. Returns nullopt
// if the value is not registered as a graph input. The name itself lives on
// GraphValue::name; this helper only enforces the invariant that a
// ModelInputValue-payload value has a corresponding entry in inputs_.
std::optional<std::string> FindInputName(const ModelGraph& graph, GraphValueId value) {
    for (const auto& input: graph.GetInputs()) {
        if (input.value == value) {
            return graph.GetValue(value).name;
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

// Copies all fields from a GraphValue into a GraphValueDesc for use in
// GetValueOutputMetadata and other spec-bearing value queries.
GraphValueDesc MakeOutputDescFromValue(const GraphValue& value) {
    return GraphValueDesc{
            .spec = value.spec,
            .payload = value.payload,
            .quantization = value.quantization,
            .name = value.name,
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
            .name = original.name,
    };

    rn.outputs.reserve(original.outputs.size());
    for (GraphValueId output: original.outputs) {
        const GraphValue& value = graph.GetValue(output);
        rn.outputs.emplace_back(
                NodeOutputDesc{
                        .payload = value.payload,
                        .quantization = value.quantization,
                        .name = value.name,
                },
                output);
    }
    return rn;
}

}// namespace

// Constructs a session over a source graph.
//
// Steps:
//   1. Size node_to_rewrite_ to the source node count.
//   2. Size value_replacements_ and resolved_value_cache_ to the source value
//      count, so every source id is in range for the tracking tables.
GraphRewriteSession::GraphRewriteSession(const ModelGraph& graph)
    : graph_(graph),
      node_to_rewrite_(graph.GetNodes().size(), std::nullopt),
      value_replacements_(graph.GetValues().size(), std::nullopt),
      resolved_value_cache_(graph.GetValues().size(), std::nullopt) {}

// Allocates a virtual value id for internal use within a subgraph replacement.
// Virtual values serve as edges between replacement nodes and are not
// persisted in the committed graph.
//
// Steps:
//   1. Take the next id in the session-local space.
//   2. Append a nullopt (virtual) entry to session_value_metadata_.
//   3. Invalidate the consumer cache and return the new id.
GraphValueId GraphRewriteSession::AllocateVirtualValue() {
    // Virtual values occupy the id space starting at graph_.GetValues().size().
    // session_value_metadata_ grows in parallel: nullopt for virtual, SessionConstant for AddSessionConstant.
    const std::size_t next_value_index = graph_.GetValues().size() + session_value_metadata_.size();
    AM_CHECK(next_value_index < std::numeric_limits<uint32_t>::max(),
             "Graph virtual value id space exhausted");
    session_value_metadata_.emplace_back(std::nullopt);
    InvalidateConsumerCache();
    return {.index = static_cast<uint32_t>(next_value_index)};
}

// Adds a new session constant value scoped to this session.
//
// Steps:
//   1. Take the next session-local id.
//   2. Record spec, binding, quantization, and name in session_value_metadata_
//      (its presence distinguishes the id from a virtual value).
//   3. Invalidate the consumer cache and return the new id.
GraphValueId GraphRewriteSession::AddSessionConstant(TensorSpec spec,
                                                     ConstantBinding binding,
                                                     QuantizationSpec quantization,
                                                     std::string name) {
    // Same id space as virtual values, but distinguished by a non-nullopt
    // SessionConstant entry in session_value_metadata_.
    const std::size_t next_value_index = graph_.GetValues().size() + session_value_metadata_.size();
    AM_CHECK(next_value_index < std::numeric_limits<uint32_t>::max(),
             "Graph session constant value id space exhausted");
    session_value_metadata_.emplace_back(SessionConstant{.spec = std::move(spec),
                                                         .binding = std::move(binding),
                                                         .quantization = quantization,
                                                         .name = std::move(name)});
    InvalidateConsumerCache();
    return {.index = static_cast<uint32_t>(next_value_index)};
}

// Applies a batch of mutations sequentially; non-atomic: if mutation N fails,
// mutations 0..N-1 remain applied to the session.
//
// Steps:
//   1. Visit each mutation in order, dispatching by variant type:
//      SubgraphReplacement -> ReplaceSubgraph, NodeRemoval -> RemoveNode,
//      InputRedirection -> RedirectInput, ValueReplacement -> ReplaceValue.
//   2. Return the first error encountered, or Status::Ok() when all mutations
//      succeed.
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

// Removes a live node. Equivalent to ReplaceSubgraph({node}, {}).
//
// Steps:
//   1. Validate the node id against the source graph.
//   2. Delegate to ReplaceSubgraph with an empty replacement list, which
//      deactivates any rewrite covering the node and records the removal.
Status GraphRewriteSession::RemoveNode(GraphNodeId node) {
    AM_RETURN_IF_ERROR(CheckSourceNodeId(node));
    const std::array old_nodes{node};
    return ReplaceSubgraph(old_nodes, {});
}

// Replaces a set of source nodes with replacement nodes.
//
// Steps:
//   1. Validate replacement node ids and targets, then replay InferOperator
//      over the replacements (semantic check) before any state mutation.
//   2. Deactivate every existing rewrite that overlaps the old_nodes.
//   3. Append a new RewriteEntry and map each old_node to it in
//      node_to_rewrite_.
//   4. Invalidate the consumer cache.
Status GraphRewriteSession::ReplaceSubgraph(std::span<const GraphNodeId> old_nodes,
                                            const std::vector<ReplacementNode>& replacement_nodes) {
    if (old_nodes.empty()) {
        return Status::InvalidArgument(
                "GraphRewriteSession::ReplaceSubgraph old node list is empty");
    }

    for (const auto& replacement: replacement_nodes) {
        AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
    }
    AM_RETURN_IF_ERROR(ValidateReplacementTargets(old_nodes, replacement_nodes));

    // Early failure: replay InferOperator over the replacement nodes before
    // mutating any session state. This catches undefined virtual inputs,
    // duplicate virtual producers, dtype mismatches, and other semantic
    // violations at Apply time rather than deferring them to Commit. The
    // scratch map is local to this replacement group; virtual values from
    // other rewrites are intentionally absent and will be rejected here.
    {
        std::vector<std::optional<TensorSpec>> virtual_specs(session_value_metadata_.size(),
                                                             std::nullopt);
        AM_RETURN_IF_ERROR(ValidateReplacementSemantics(replacement_nodes, virtual_specs));
    }

    for (const auto [index]: old_nodes) {
        if (const auto rewrite_index = node_to_rewrite_[index];
            rewrite_index.has_value()) {
            DeactivateRewrite(*rewrite_index);
        }
    }

    const std::size_t rewrite_index = rewrites_.size();
    rewrites_.emplace_back(std::vector<GraphNodeId>{old_nodes.begin(), old_nodes.end()},
                           replacement_nodes, true, false);

    for (const auto [index]: old_nodes) {
        node_to_rewrite_[index] = rewrite_index;
    }
    InvalidateConsumerCache();
    return Status::Ok();
}

// Rewires one input of a live node to a different value.
//
// Steps:
//   1. Validate the node id, the non-virtual new_value, and the input index
//      range.
//   2. If the node already has a rewrite entry: mutate that replacement's
//      input in place (rejecting non-live nodes).
//   3. Otherwise: build a mirror replacement of the original node, apply the
//      input change, and append a rewrite with exposes_node_view set so the
//      original node identity stays live.
//   4. Invalidate the consumer cache.
Status GraphRewriteSession::RedirectInput(GraphNodeId node, size_t input_index,
                                          GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckSourceNodeId(node));
    AM_RETURN_IF_ERROR(CheckNonVirtualValueId(new_value));
    if (input_index >= graph_.GetNode(node).inputs.size()) {
        return Status::InvalidArgument(
                "GraphRewriteSession::RedirectInput input index out of range");
    }

    // The redirected value must keep the slot's dtype: the node's op is fixed,
    // so a dtype-changing redirect would only fail later in the InferOperator
    // replay during ValidateEdits/Commit. Reject it here with exact slot
    // context. Shape compatibility is deliberately deferred to
    // committed.Validate().
    const GraphValueId old_value = graph_.GetNode(node).inputs[input_index];
    const TensorSpec old_spec = graph_.GetValue(old_value).spec;
    const StatusOr<TensorSpec> new_spec_or = ResolveValueSpec(GetResolvedValue(new_value), {});
    if (!new_spec_or.ok()) {
        return Status::InvalidArgument(
                "GraphRewriteSession::RedirectInput new value " +
                std::to_string(new_value.index) +
                " resolves to a session virtual value, which cannot feed a node input");
    }

    if (new_spec_or->dtype != old_spec.dtype) {
        return Status::InvalidArgument(
                "GraphRewriteSession::RedirectInput input dtype mismatch: node '" +
                graph_.GetNode(node).name + "' input " + std::to_string(input_index) +
                " expects " + ToString(old_spec.dtype) + " but new value " +
                std::to_string(new_value.index) + " is " +
                ToString(new_spec_or->dtype));
    }

    // IsNodeLive returns true both for untouched nodes and for nodes with a
    // live single-node mirror rewrite. Only the latter can be mutated in place,
    // so guard with has_value() to exclude the untouched-node case.
    if (const auto existing = node_to_rewrite_[node.index]; existing.has_value()) {
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
    rewrites_.emplace_back(std::vector<GraphNodeId>{node},
                           std::vector<ReplacementNode>{std::move(replacement)},
                           true,
                           true);
    node_to_rewrite_[node.index] = idx;
    InvalidateConsumerCache();
    return Status::Ok();
}

// Redirects consumers of `old_value` to `new_value` after resolution.
//
// Steps:
//   1. Validate that old_value is a source value and new_value is non-virtual;
//      early-out when both ids are equal.
//   2. Walk new_value's replacement chain and reject the edit if it reaches
//      old_value (would close a cycle).
//   3. Record value_replacements_[old_value] = new_value.
//   4. Clear the resolved value cache and invalidate the consumer cache.
Status GraphRewriteSession::ReplaceValue(GraphValueId old_value, GraphValueId new_value) {
    AM_RETURN_IF_ERROR(CheckSourceValueId(old_value));
    AM_RETURN_IF_ERROR(CheckNonVirtualValueId(new_value));
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

// Walks the replacement chain from `value` to its terminal, with path
// compression: all values along the resolution path are cached to the
// terminal for O(1) subsequent lookups.
//
// Steps:
//   1. Return the cached terminal when present; out-of-range ids return
//      identity.
//   2. Walk value_replacements_ from `value`, recording the visited path,
//      until a terminal value or a cached entry is reached.
//   3. Cache every value on the path to the terminal (path compression) and
//      return the terminal.
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

// Returns a snapshot of a live node with inputs resolved through
// GetResolvedValue; outputs are the original graph value ids.
//
// Steps:
//   1. Validate the node id and locate the covering rewrite via
//      node_to_rewrite_; reject non-live nodes.
//   2. Assemble the view from the RedirectInput mirror replacement when
//      present, otherwise from the original node; outputs always come from
//      the original node.
//   3. Resolve every input through GetResolvedValue.
StatusOr<GraphNodeView> GraphRewriteSession::GetNodeView(GraphNodeId node) const {
    AM_RETURN_IF_ERROR(CheckSourceNodeId(node));

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
            .name = replacement ? replacement->name : original.name,
    };

    for (GraphValueId& input: view.inputs) {
        input = GetResolvedValue(input);
    }
    return view;
}

// Returns true if `node` is currently observable in the session.
//
// Steps:
//   1. Out-of-range ids are not live.
//   2. Ids without a rewrite entry are live (untouched nodes).
//   3. A rewrite entry is live only when it is active, exposes a node view,
//      covers exactly this single node, and has exactly one replacement (a
//      RedirectInput mirror).
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

// Returns live node ids in topological order, following the original graph's
// ordering.
//
// Steps:
//   1. Obtain the source graph's topological order.
//   2. Filter it with IsNodeLive, preserving the source ordering.
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

// Returns live node ids whose op_type matches `op_type`, in ascending
// node-index order.
//
// Steps:
//   1. Query the source graph for nodes with matching op_types.
//   2. Filter the result with IsNodeLive.
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

// Returns true if `value` is structurally present in the current session view.
//
// Steps:
//   1. Session constants are live.
//   2. Out-of-range ids are not live.
//   3. Source values without a producer (external values) are live.
//   4. Values produced by a live node are live.
//   5. Otherwise: live only when an active rewrite takes over the value via a
//      replacement output's `replaces` binding.
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

// Returns the spec-bearing descriptor for an existing graph value.
//
// Steps:
//   1. Reject virtual and out-of-range ids.
//   2. Build the descriptor from the session constant's metadata for session
//      constants, otherwise from the source graph value.
StatusOr<GraphValueDesc> GraphRewriteSession::GetValueOutputMetadata(GraphValueId value) const {
    AM_RETURN_IF_ERROR(CheckNonVirtualValueId(value));
    if (IsSessionConstant(value)) {
        return MakeOutputDescFromSessionConstant(value);
    }
    return MakeOutputDescFromValue(graph_.GetValue(value));
}

// Returns true when `value` is directly marked as a graph output in the
// source graph.
//
// Steps:
//   1. Reject out-of-range ids.
//   2. Linearly scan the source graph's output list for `value`.
bool GraphRewriteSession::IsGraphOutput(GraphValueId value) const noexcept {
    if (value.index >= graph_.GetValues().size()) {
        return false;
    }

    return std::ranges::any_of(graph_.GetOutputs(), [&](const auto& output) {
        return output.value == value;
    });
}

// True when any active rewrite's replacement output takes over this value.
//
// Steps:
//   1. Iterate over active rewrites and their replacement node outputs.
//   2. Return true when an output's `replaces` binding equals `value`.
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

// Returns all live value ids in ascending index order.
//
// Steps:
//   1. Iterate source graph values, appending ids that satisfy IsValueLive.
//   2. Iterate session-local values, appending ids that satisfy IsValueLive.
std::vector<GraphValueId> GraphRewriteSession::GetLiveValues() const {
    const std::span<const GraphValue> values = graph_.GetValues();
    std::vector<GraphValueId> live;
    live.reserve(values.size() + session_value_metadata_.size());
    for (uint32_t i = 0; i < values.size(); ++i) {
        if (const GraphValueId id{.index = i}; IsValueLive(id)) {
            live.push_back(id);
        }
    }
    for (uint32_t i = 0; i < session_value_metadata_.size(); ++i) {
        const GraphValueId id{.index = static_cast<uint32_t>(values.size() + i)};
        if (IsValueLive(id)) {
            live.push_back(id);
        }
    }
    return live;
}

// Returns live original graph nodes that consume `value` (after resolution)
// as an input, in topological order.
//
// Steps:
//   1. Reject virtual values and unknown ids.
//   2. Resolve `value` through any ReplaceValue chain.
//   3. Return the cached consumer bucket for the resolved id (built by
//      EnsureConsumerCache).
StatusOr<std::vector<GraphNodeId>> GraphRewriteSession::FindConsumers(GraphValueId value) const {
    if (value.index >= graph_.GetValues().size() && !IsSessionLocalValue(value)) {
        return std::vector<GraphNodeId>{};
    }

    if (IsSessionVirtualValue(value)) {
        return std::vector<GraphNodeId>{};
    }

    const GraphValueId resolved_value = GetResolvedValue(value);
    AM_ASSIGN_OR_RETURN(const ConsumerCache* cache, EnsureConsumerCache());
    if (resolved_value.index >= cache->original_consumers.size()) {
        return std::vector<GraphNodeId>{};
    }
    return cache->original_consumers[resolved_value.index];
}

// Returns true if any live node or active replacement node consumes `value`
// (after resolution) as an input.
//
// Steps:
//   1. Reject virtual values and unknown ids.
//   2. Resolve `value` through any ReplaceValue chain.
//   3. Consult the consumer cache: true when the resolved value has any live
//      original consumer or a positive replacement consumer count.
StatusOr<bool> GraphRewriteSession::HasLiveConsumers(GraphValueId value) const {
    if (value.index >= graph_.GetValues().size() && !IsSessionLocalValue(value)) {
        return false;
    }

    if (IsSessionVirtualValue(value)) {
        return false;
    }

    const GraphValueId resolved_value = GetResolvedValue(value);
    AM_ASSIGN_OR_RETURN(const ConsumerCache* cache, EnsureConsumerCache());
    if (resolved_value.index >= cache->original_consumers.size()) {
        return false;
    }
    return !cache->original_consumers[resolved_value.index].empty() ||
           cache->replacement_consumer_counts[resolved_value.index] > 0;
}

// Builds or returns the consumer index for the current mutation generation.
//
// Steps:
//   1. Return the cached index when its generation matches
//      mutation_generation_.
//   2. Otherwise rebuild: live nodes in topological order (inputs resolved via
//      GetResolvedValue, mirror inputs for RedirectInput'd nodes) populate
//      original_consumers.
//   3. Count active non-mirror replacement inputs into
//      replacement_consumer_counts.
//
// Returns InvalidArgument if the source graph contains a cycle; the session
// does not introduce new edges, so a cycle can only originate from the source
// graph. On failure the previous cache (if any) is left intact.
StatusOr<const GraphRewriteSession::ConsumerCache*> GraphRewriteSession::EnsureConsumerCache() const {
    if (consumer_cache_.has_value() && consumer_cache_->generation == mutation_generation_) {
        return &*consumer_cache_;
    }

    ConsumerCache cache;
    cache.generation = mutation_generation_;
    const std::size_t value_count = graph_.GetValues().size() + session_value_metadata_.size();
    cache.original_consumers.resize(value_count);
    cache.replacement_consumer_counts.resize(value_count, 0U);

    const StatusOr<std::vector<GraphNodeId>> order = graph_.TopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());

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
                const auto resolved_input = GetResolvedValue(input);
                if (resolved_input.index < value_count) {
                    ++cache.replacement_consumer_counts[resolved_input.index];
                }
            }
        }
    }

    consumer_cache_ = std::move(cache);
    return &*consumer_cache_;
}

// Validates the session's internal consistency without materializing.
//
// Steps:
//   1. Validate every installed ReplaceValue target as non-virtual.
//   2. For each active rewrite, run ValidateReplacementNode and
//      ValidateReplacementTargets.
//   3. Run ValidateVirtualValues.
//   4. Replay ValidateReplacementSemantics per active rewrite.
Status GraphRewriteSession::ValidateEdits() const {
    for (const auto& replacement: value_replacements_) {
        if (replacement.has_value()) {
            AM_RETURN_IF_ERROR(CheckNonVirtualValueId(*replacement));
        }
    }

    for (const auto& rewrite: rewrites_) {
        if (!rewrite.active) {
            continue;
        }

        for (const auto& replacement: rewrite.replacements) {
            AM_RETURN_IF_ERROR(ValidateReplacementNode(replacement));
        }
        AM_RETURN_IF_ERROR(ValidateReplacementTargets(rewrite.old_nodes, rewrite.replacements));
    }
    AM_RETURN_IF_ERROR(ValidateVirtualValues());

    // Semantic replay: validate each active rewrite by replaying InferOperator
    // over its replacement nodes. Virtual value specs are derived per-rewrite
    // (ValidateVirtualValues ensures virtual values don't cross rewrite
    // boundaries, so a fresh scratch map per rewrite is correct).
    for (const auto& rewrite: rewrites_) {
        if (!rewrite.active) {
            continue;
        }
        std::vector<std::optional<TensorSpec>> virtual_specs(session_value_metadata_.size(),
                                                             std::nullopt);
        AM_RETURN_IF_ERROR(ValidateReplacementSemantics(rewrite.replacements, virtual_specs));
    }

    // Emission-order availability: every replacement input must be produced
    // before its rewrite's commit emission point. Without this, Commit would
    // fail with an unmappable-input error after ValidateEdits already passed.
    AM_RETURN_IF_ERROR(ValidateReplacementInputAvailability());
    return Status::Ok();
}

GraphValueDesc GraphRewriteSession::MakeOutputDescFromSessionConstant(GraphValueId value) const {
    const SessionConstant& constant = *session_value_metadata_[GetSessionValueIndex(value)];
    return {.spec = constant.spec,
            .payload = ConstantValue{.binding = constant.binding},
            .quantization = constant.quantization,
            .name = constant.name};
}

// Translates a value id from source/session space to committed graph space.
//
// Steps:
//   1. Source values: map through source_values.
//   2. Session constants: map through session_constants.
//   3. Virtual values: map through virtual_values.
//   4. Unmapped or out-of-range ids yield InvalidArgument.
StatusOr<GraphValueId> GraphRewriteSession::MapCommittedValue(
        GraphValueId value,
        const CommitValueMaps& maps) const {
    if (value.index < graph_.GetValues().size()) {
        return MapResolvedValue(value, maps.source_values);
    }

    const std::size_t session_index = GetSessionValueIndex(value);
    if (session_index >= session_value_metadata_.size()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: session value id out of range during commit");
    }

    if (IsSessionConstant(value)) {
        if (session_index >= maps.session_constants.size() ||
            !maps.session_constants[session_index].has_value()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession: session constant cannot be mapped during commit");
        }
        return *maps.session_constants[session_index];
    }

    if (session_index >= maps.virtual_values.size() ||
        !maps.virtual_values[session_index].has_value()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: virtual value " + std::to_string(value.index) +
                " cannot be mapped during commit (not produced within its rewrite)");
    }
    return *maps.virtual_values[session_index];
}

// Copies source external values and session constants into the committed
// graph.
//
// Steps:
//   1. Add each producer-less source value as input, weight, constant, or
//      state per its payload (monostate/unsupported payloads error).
//   2. Apply quantization to each added value.
//   3. Add every session constant via committed.AddConstant.
Status GraphRewriteSession::CopyExternalValues(ModelGraph& committed,
                                               const CommitValueMaps& maps) const {
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
            maps.source_values[i] = committed.AddWeight(value.spec, weight->binding, value.name);
        } else if (const auto* constant = std::get_if<ConstantValue>(&value.payload)) {
            maps.source_values[i] = committed.AddConstant(value.spec, constant->binding, value.name);
        } else if (const auto* state = std::get_if<StateValue>(&value.payload)) {
            maps.source_values[i] = committed.AddState(value.spec, state->binding, value.name);
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

    for (uint32_t i = 0; i < session_value_metadata_.size(); ++i) {
        if (!session_value_metadata_[i].has_value()) {
            continue;
        }

        const SessionConstant& constant = *session_value_metadata_[i];
        maps.session_constants[i] = committed.AddConstant(constant.spec,
                                                          constant.binding,
                                                          constant.name);
        committed.SetQuantization(*maps.session_constants[i], constant.quantization);
    }
    return Status::Ok();
}

// Emits all replacement nodes in a rewrite entry into the committed graph.
//
// Steps (per replacement node):
//   1. Resolve and map its inputs into the committed value space.
//   2. Add the node to the committed graph.
//   3. Map each output's `replaces` target into the appropriate value map
//      (virtual or source); session-constant targets and double mappings are
//      rejected.
Status GraphRewriteSession::EmitRewrite(const RewriteEntry& rewrite,
                                        ModelGraph& committed,
                                        const CommitValueMaps& maps) const {
    // For each replacement node, resolve and map inputs, add the node to
    // the committed graph, then map each output through the replaces binding
    // into the appropriate map (source_values, virtual_values, or error).
    for (const auto& replacement: rewrite.replacements) {
        std::vector<GraphValueId> new_inputs;
        new_inputs.reserve(replacement.inputs.size());
        for (const auto input: replacement.inputs) {
            const auto resolved_input = GetResolvedValue(input);
            auto mapped_input = MapCommittedValue(resolved_input, maps);
            AM_RETURN_IF_ERROR(mapped_input.status());
            new_inputs.push_back(*mapped_input);
        }

        std::vector<NodeOutputDesc> output_descs;
        output_descs.reserve(replacement.outputs.size());
        for (const auto& output: replacement.outputs) {
            output_descs.push_back(output.desc);
        }

        const auto added_or = committed.AddNode(
                replacement.op_type,
                replacement.decoder_layer_index,
                std::move(new_inputs),
                std::move(output_descs),
                replacement.op_params,
                replacement.attrs,
                replacement.name);
        AM_RETURN_IF_ERROR(added_or.status());
        const AddedNode& added = *added_or;

        for (size_t i = 0; i < replacement.outputs.size(); ++i) {
            if (replacement.outputs[i].replaces.has_value()) {
                if (const GraphValueId replaced = *replacement.outputs[i].replaces;
                    IsSessionVirtualValue(replaced)) {
                    if (maps.virtual_values[GetSessionValueIndex(replaced)].has_value()) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession::Commit replacement virtual value was already mapped");
                    }
                    maps.virtual_values[GetSessionValueIndex(replaced)] = added.outputs[i];
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

// Emits a single surviving original node into the committed graph.
//
// Steps:
//   1. Snapshot the node via GetNodeView.
//   2. Resolve and map its inputs into the committed value space.
//   3. Add the node to the committed graph.
//   4. Map each original output value into source_values, rejecting
//      already-mapped outputs.
Status GraphRewriteSession::EmitOriginalNode(GraphNodeId old_node,
                                             ModelGraph& committed,
                                             const CommitValueMaps& maps) const {
    // Emit a surviving original node (untouched or RedirectInput'd) into the
    // committed graph. Uses GetNodeView to get the resolved input view and
    // faithfully reproduces the node's outputs.
    StatusOr<GraphNodeView> view = GetNodeView(old_node);
    AM_RETURN_IF_ERROR(view.status());

    std::vector<GraphValueId> new_inputs;
    new_inputs.reserve(view->inputs.size());
    for (const auto input: view->inputs) {
        auto mapped_input = MapCommittedValue(input, maps);
        AM_RETURN_IF_ERROR(mapped_input.status());
        new_inputs.push_back(*mapped_input);
    }

    std::vector<NodeOutputDesc> output_descs;
    output_descs.reserve(view->outputs.size());
    for (const auto old_output: view->outputs) {
        const auto& old_value = graph_.GetValue(old_output);
        output_descs.push_back({
                .payload = old_value.payload,
                .quantization = old_value.quantization,
                .name = old_value.name,
        });
    }

    const auto added_or = committed.AddNode(
            view->op_type,
            view->decoder_layer_index,
            std::move(new_inputs),
            std::move(output_descs),
            view->op_params,
            view->attrs,
            view->name);
    AM_RETURN_IF_ERROR(added_or.status());
    const AddedNode& added = *added_or;

    for (size_t i = 0; i < view->outputs.size(); ++i) {
        if (maps.source_values[view->outputs[i].index].has_value()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession::Commit original node output was already mapped");
        }
        maps.source_values[view->outputs[i].index] = added.outputs[i];
    }
    return Status::Ok();
}

// Maps graph output values through resolution into the committed graph.
//
// Steps:
//   1. For each source graph output, resolve it through any ReplaceValue
//      chain.
//   2. Map it into the committed value space and mark it via
//      committed.MarkOutput.
Status GraphRewriteSession::MarkCommittedOutputs(ModelGraph& committed,
                                                 const CommitValueMaps& maps) const {
    // Graph outputs are resolved through ReplaceValue chains before mapping
    // into the committed graph's value space.
    for (const auto& [value]: graph_.GetOutputs()) {
        const auto resolved_output = GetResolvedValue(value);
        auto mapped_output = MapCommittedValue(resolved_output, maps);
        AM_RETURN_IF_ERROR(mapped_output.status());
        committed.MarkOutput(*mapped_output);
    }
    return Status::Ok();
}

// Materializes the session state into a new ModelGraph.
//
// Steps:
//   1. Validate the session with ValidateEdits().
//   2. Copy external values and session constants (CopyExternalValues).
//   3. Traverse the source graph in topological order, emitting each rewrite
//      at its first old_node and emitting surviving original nodes.
//   4. Mark the committed graph outputs (MarkCommittedOutputs) and validate
//      the result.
StatusOr<ModelGraph> GraphRewriteSession::Commit() const {
    AM_RETURN_IF_ERROR(ValidateEdits());

    ModelGraph committed;
    ValueMap value_map(graph_.GetValues().size(), std::nullopt);
    ValueMap session_constant_map(session_value_metadata_.size(), std::nullopt);
    ValueMap virtual_value_map(session_value_metadata_.size(), std::nullopt);
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

// Returns Ok when `node` is a valid source graph node id.
//
// Steps:
//   1. Compare the id against the source graph node count.
//   2. Out-of-range ids yield InvalidArgument.
Status GraphRewriteSession::CheckSourceNodeId(GraphNodeId node) const {
    if (node.index >= graph_.GetNodes().size()) {
        return Status::InvalidArgument(
                "GraphRewriteSession: source node id out of range");
    }
    return Status::Ok();
}

// Rejects value ids that are not addressable source/session values.
//
// Steps:
//   1. Classify the value id.
//   2. kSource and kSessionConstant ids pass.
//   3. kSessionVirtual ids fail with a virtual-value diagnostic; anything
//      unbounded fails with an out-of-range diagnostic.
Status GraphRewriteSession::CheckNonVirtualValueId(GraphValueId value) const {
    const ValueKind kind = ClassifyValue(value);
    if (kind == ValueKind::kSource || kind == ValueKind::kSessionConstant) {
        return Status::Ok();
    }

    if (kind == ValueKind::kSessionVirtual) {
        return Status::InvalidArgument(
                "GraphRewriteSession: virtual value is not allowed in this operation");
    }
    return Status::InvalidArgument("GraphRewriteSession: value id out of range");
}

// Requires the value id to name a value from the source graph.
//
// Steps:
//   1. Classify the value id.
//   2. kSource ids pass.
//   3. kSessionConstant and kSessionVirtual ids fail with a kind-specific
//      diagnostic; unbounded ids fail with out-of-range.
Status GraphRewriteSession::CheckSourceValueId(GraphValueId value) const {
    const ValueKind kind = ClassifyValue(value);
    if (kind == ValueKind::kSource) {
        return Status::Ok();
    }

    if (kind == ValueKind::kSessionConstant) {
        return Status::InvalidArgument(
                "GraphRewriteSession: expected source value id, got session constant");
    }

    if (kind == ValueKind::kSessionVirtual) {
        return Status::InvalidArgument(
                "GraphRewriteSession: expected source value id, got session virtual value");
    }
    return Status::InvalidArgument("GraphRewriteSession: value id out of range");
}

// Returns Ok for any value id the session knows (source, session constant,
// or session virtual).
//
// Steps:
//   1. Classify the value id.
//   2. Any kind other than kInvalid passes; kInvalid yields out-of-range.
Status GraphRewriteSession::CheckKnownValueId(GraphValueId value) const {
    if (ClassifyValue(value) != ValueKind::kInvalid) {
        return Status::Ok();
    }
    return Status::InvalidArgument("GraphRewriteSession: value id out of range");
}

// Classifies a value id into the session's value-kind space.
//
// Steps:
//   1. Ids below the source graph value range are kSource.
//   2. Ids in the session-local range are kSessionConstant when
//      session_value_metadata_ holds a constant, otherwise kSessionVirtual.
//   3. Ids beyond the session-local range are kInvalid (never allocated).
GraphRewriteSession::ValueKind GraphRewriteSession::ClassifyValue(
        GraphValueId value) const noexcept {
    if (value.index < graph_.GetValues().size()) {
        return ValueKind::kSource;
    }

    const std::size_t session_index = GetSessionValueIndex(value);
    if (session_index >= session_value_metadata_.size()) {
        return ValueKind::kInvalid;
    }

    return session_value_metadata_[session_index].has_value()
                   ? ValueKind::kSessionConstant
                   : ValueKind::kSessionVirtual;
}

bool GraphRewriteSession::IsSessionLocalValue(GraphValueId value) const noexcept {
    const ValueKind kind = ClassifyValue(value);
    return kind == ValueKind::kSessionConstant || kind == ValueKind::kSessionVirtual;
}

bool GraphRewriteSession::IsSessionConstant(GraphValueId value) const noexcept {
    return ClassifyValue(value) == ValueKind::kSessionConstant;
}

bool GraphRewriteSession::IsSessionVirtualValue(GraphValueId value) const noexcept {
    return ClassifyValue(value) == ValueKind::kSessionVirtual;
}

// Returns true if `value` resolves to a compile-time constant.
//
// Steps:
//   1. Resolve the value through any ReplaceValue chain to its terminal.
//   2. Return true for session constants and for source values carrying a
//      ConstantValue or WeightValue payload; virtual and out-of-range values
//      return false.
bool GraphRewriteSession::IsConstant(GraphValueId value) const {
    const GraphValueId resolved = GetResolvedValue(value);

    // Session constants (added via session.AddSessionConstant).
    if (IsSessionConstant(resolved)) {
        return true;
    }

    if (resolved.index < graph_.GetValues().size()) {
        const GraphValuePayload& payload = graph_.GetValue(resolved).payload;
        return std::holds_alternative<ConstantValue>(payload) ||
               std::holds_alternative<WeightValue>(payload);
    }

    // Virtual values / out-of-range → not constant.
    return false;
}

// Returns true if all resolved inputs of `node` are compile-time constants.
//
// Steps:
//   1. Snapshot the node via GetNodeView; a non-live node fails the status
//      check and yields false.
//   2. Apply IsConstant to every resolved input; no inputs is vacuously true.
bool GraphRewriteSession::AreAllInputsConstant(GraphNodeId node) const {
    const auto view = GetNodeView(node);
    if (!view.ok()) {
        return false;
    }

    return std::ranges::all_of(view->inputs, [&](const auto& input) {
        return IsConstant(input);
    });
}

// Validates that every reference inside a replacement node points to a
// known value id.
//
// Steps:
//   1. Check each replacement input via CheckKnownValueId.
//   2. Check each `replaces` binding (when present) via CheckKnownValueId.
Status GraphRewriteSession::ValidateReplacementNode(
        const ReplacementNode& replacement) const {
    for (auto input: replacement.inputs) {
        AM_RETURN_IF_ERROR(CheckKnownValueId(input));
    }

    for (const auto& output: replacement.outputs) {
        if (output.replaces.has_value()) {
            AM_RETURN_IF_ERROR(CheckKnownValueId(*output.replaces));
        }
    }
    return Status::Ok();
}

// Validates that replacement targets belong to old_nodes and are not
// duplicated.
//
// Steps:
//   1. Collect the outputs of every old_node.
//   2. For each replacement output's `replaces` target (virtual targets
//      excluded), reject targets not produced by the old_nodes or produced
//      more than once.
Status GraphRewriteSession::ValidateReplacementTargets(
        std::span<const GraphNodeId> old_nodes,
        const std::vector<ReplacementNode>& replacement_nodes) const {
    std::vector<GraphValueId> replaceable_outputs;
    for (const auto old_node: old_nodes) {
        AM_RETURN_IF_ERROR(CheckSourceNodeId(old_node));
        const auto& node = graph_.GetNode(old_node);
        replaceable_outputs.insert(replaceable_outputs.end(),
                                   node.outputs.begin(), node.outputs.end());
    }

    std::vector<GraphValueId> real_replacements;
    for (const auto& replacement: replacement_nodes) {
        for (const auto& [_, replaces]: replacement.outputs) {
            if (!replaces.has_value() || IsSessionVirtualValue(*replaces)) {
                continue;
            }

            const GraphValueId replaced = *replaces;
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

// Validates virtual value ordering (no consumption before production).
//
// Steps:
//   1. Track virtual values produced by any active rewrite (global set);
//      reject duplicate production of the same virtual value.
//   2. Per rewrite, track virtual values available so far; reject virtual
//      inputs consumed before being produced within the group.
Status GraphRewriteSession::ValidateVirtualValues() const {
    // Two-level tracking: globally_produced prevents duplicate production of
    // any virtual value across all rewrites. locally_available (per rewrite)
    // ensures each virtual value is produced before it is consumed within the
    // same rewrite group.
    std::vector globally_produced(session_value_metadata_.size(), false);

    for (const auto& rewrite: rewrites_) {
        if (!rewrite.active) {
            continue;
        }

        std::vector locally_available(session_value_metadata_.size(), false);
        for (const auto& replacement: rewrite.replacements) {
            for (GraphValueId input: replacement.inputs) {
                if (IsSessionVirtualValue(input)) {
                    if (!locally_available[GetSessionValueIndex(input)]) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession: virtual value is consumed before being produced");
                    }
                }
            }

            for (const auto& output: replacement.outputs) {
                if (!output.replaces.has_value() || !IsSessionVirtualValue(*output.replaces)) {
                    continue;
                }

                const std::size_t virtual_index = GetSessionValueIndex(*output.replaces);
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

// Resolves the TensorSpec of a value id known to the session.
//
// Steps:
//   1. Source values: return the spec stored in graph_.
//   2. Session constants: return the spec stored in session_value_metadata_.
//   3. Virtual values: return the inferred spec in virtual_specs
//      (caller-provided scratch map), or NotFound when absent.
StatusOr<TensorSpec> GraphRewriteSession::ResolveValueSpec(
        GraphValueId value,
        const std::vector<std::optional<TensorSpec>>& virtual_specs) const {
    // Source graph value: read spec directly from the source graph.
    if (value.index < graph_.GetValues().size()) {
        return graph_.GetValue(value).spec;
    }

    // Out-of-range check (defense in depth; ValidateReplacementNode should
    // have already rejected invalid ids via CheckKnownValueId).
    if (!IsSessionLocalValue(value)) {
        return Status::InvalidArgument(
                "GraphRewriteSession::ResolveValueSpec: value id " +
                std::to_string(value.index) + " out of range");
    }

    // Session constant: spec is stored in session_value_metadata_.
    if (IsSessionConstant(value)) {
        return session_value_metadata_[GetSessionValueIndex(value)]->spec;
    }

    // Virtual value: must have an inferred spec in the caller's scratch map.
    // A virtual value without an inferred spec has not been produced by any
    // analyzed replacement, so its dtype/shape are unknown.
    const std::size_t idx = GetSessionValueIndex(value);
    if (idx >= virtual_specs.size() || !virtual_specs[idx].has_value()) {
        return Status::NotFound(
                "GraphRewriteSession: virtual value " + std::to_string(value.index) +
                " has no inferred spec (not produced by any analyzed replacement)");
    }
    return *virtual_specs[idx];
}

// Replays InferOperator over replacement nodes in order, verifying input
// spec availability (dtype/shape resolvability), output count, replaces
// target dtype compatibility, and deriving virtual specs into
// virtual_specs_out. Emission-order availability of source value inputs is
// NOT checked here; ValidateReplacementInputAvailability handles that.
//
// Steps (per replacement):
//   1. Look up the operator schema.
//   2. Check input count against the schema.
//   3. Check output count against the schema.
//   4. Resolve input specs (through ReplaceValue chains).
//   5. Replay InferOperator to derive output specs.
//   6. Check that the inferred output count matches the binding count.
//   7. Per output binding: populate the virtual spec scratch map or verify
//      dtype compatibility with the replaced target.
Status GraphRewriteSession::ValidateReplacementSemantics(
        const std::vector<ReplacementNode>& replacements,
        std::vector<std::optional<TensorSpec>>& virtual_specs_out) const {
    for (std::size_t i = 0; i < replacements.size(); ++i) {
        const auto& replacement = replacements[i];
        const std::string debug_ctx =
                "replacement[" + std::to_string(i) + "]" +
                (replacement.name.empty() ? "" : " '" + replacement.name + "'");

        // 1. Schema lookup: confirms op_type is registered and gives us the
        //    expected input/output port counts. Parameter (variant) validation
        //    happens later inside the InferOperator replay.
        AM_ASSIGN_OR_RETURN(const OperatorSchema schema,
                            GetOperatorSchema(replacement.op_type));

        // 2. Input count must match schema.
        if (replacement.inputs.size() != schema.input_ports.size()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession: " + debug_ctx + " input count " +
                    std::to_string(replacement.inputs.size()) +
                    " does not match schema expected " +
                    std::to_string(schema.input_ports.size()));
        }

        // 3. Output count must match schema.
        if (replacement.outputs.size() != schema.output_ports.size()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession: " + debug_ctx + " output count " +
                    std::to_string(replacement.outputs.size()) +
                    " does not match schema expected " +
                    std::to_string(schema.output_ports.size()));
        }

        // 4. Resolve input specs. Virtual inputs must already be in the scratch
        //    map (i.e. produced by an earlier replacement within this replay).
        //    ReplaceValue chains are resolved first so that aliased values map
        //    to the terminal spec.
        std::vector<TensorSpec> input_specs;
        input_specs.reserve(replacement.inputs.size());
        for (std::size_t j = 0; j < replacement.inputs.size(); ++j) {
            const GraphValueId resolved = GetResolvedValue(replacement.inputs[j]);
            AM_ASSIGN_OR_RETURN(TensorSpec spec,
                                ResolveValueSpec(resolved, virtual_specs_out));
            input_specs.push_back(std::move(spec));
        }

        // 5. Replay InferOperator to derive the inferred output specs.
        //    Failures here indicate incompatible input dtypes/shapes for this
        //    operator.
        AM_ASSIGN_OR_RETURN(InferenceResult inferred,
                            InferOperator(replacement.op_type,
                                          replacement.op_params,
                                          input_specs));

        // 6. InferOperator must produce exactly one output spec per binding.
        if (inferred.outputs.size() != replacement.outputs.size()) {
            return Status::InvalidArgument(
                    "GraphRewriteSession: " + debug_ctx + " inferred " +
                    std::to_string(inferred.outputs.size()) +
                    " outputs but binding has " +
                    std::to_string(replacement.outputs.size()));
        }

        // 7. For each output binding, either populate the scratch map (virtual
        //    target) or verify dtype compatibility (source/session constant
        //    target). Shape compatibility is deferred to committed.Validate().
        for (std::size_t j = 0; j < replacement.outputs.size(); ++j) {
            const RewriteOutputBinding& binding = replacement.outputs[j];
            const TensorSpec& inferred_spec = inferred.outputs[j];

            if (!binding.replaces.has_value()) {
                continue;
            }

            const GraphValueId replaced = *binding.replaces;

            if (IsSessionVirtualValue(replaced)) {
                const std::size_t vidx = GetSessionValueIndex(replaced);
                if (vidx >= virtual_specs_out.size()) {
                    return Status::InvalidArgument(
                            "GraphRewriteSession: " + debug_ctx + " output " +
                            std::to_string(j) + " replaces out-of-range virtual value " +
                            std::to_string(replaced.index));
                }

                if (virtual_specs_out[vidx].has_value()) {
                    return Status::InvalidArgument(
                            "GraphRewriteSession: " + debug_ctx + " output " +
                            std::to_string(j) + " replaces virtual value " +
                            std::to_string(replaced.index) +
                            " which is already produced by another replacement");
                }
                virtual_specs_out[vidx] = inferred_spec;
                continue;
            }

            // Source value or session constant target: dtype must match.
            TensorSpec target_spec;
            if (replaced.index < graph_.GetValues().size()) {
                target_spec = graph_.GetValue(replaced).spec;
            } else if (IsSessionConstant(replaced)) {
                target_spec = session_value_metadata_[GetSessionValueIndex(replaced)]->spec;
            } else {
                return Status::InvalidArgument(
                        "GraphRewriteSession: " + debug_ctx + " output " +
                        std::to_string(j) + " replaces invalid value " +
                        std::to_string(replaced.index));
            }

            if (inferred_spec.dtype != target_spec.dtype) {
                return Status::InvalidArgument(
                        "GraphRewriteSession: " + debug_ctx + " output " +
                        std::to_string(j) + " inferred dtype " +
                        ToString(inferred_spec.dtype) +
                        " is incompatible with replaces target value " +
                        std::to_string(replaced.index) + " dtype " +
                        ToString(target_spec.dtype));
            }
        }
    }
    return Status::Ok();
}

// Validates that every active rewrite's replacement inputs are available at
// the rewrite's commit emission point.
//
// Commit emits each rewrite when the first old_node appears in source
// topological order (see Commit). A replacement input referencing a value
// whose producer emits later would fail to map during commit
// ("value X cannot be mapped during commit"). This check rejects such
// rewrites in ValidateEdits, so a passing ValidateEdits guarantees Commit
// cannot fail on unmappable replacement inputs.
//
// Rules (per replacement input, resolved through ReplaceValue chains):
//   1. Session constants and virtual values are always available: constants
//      are pre-mapped by CopyExternalValues; virtual values are produced
//      within the rewrite and validated by ValidateReplacementSemantics.
//   2. Source values without a producer (inputs, weights, constants, states)
//      are pre-mapped by CopyExternalValues; always available.
//   3. Source value with producer p:
//      - p untouched: available iff p's topo position is before this
//        rewrite's emission point.
//      - p covered by an active rewrite R_p that binds the value as a
//        replacement output: available iff R_p's emission point (min topo
//        position over its old_nodes) is before this rewrite's emission
//        point. A covering rewrite that never binds the value (e.g.
//        RemoveNode installs a rewrite with no replacements) means the
//        value is never emitted; rejected unconditionally.
//      - p inside this rewrite: available iff an earlier replacement in the
//        same group binds the value (EmitRewrite emits replacements in
//        array order); a later or self binding cannot be seen yet.
//      Otherwise InvalidArgument.
Status GraphRewriteSession::ValidateReplacementInputAvailability() const {
    if (rewrites_.empty()) {
        return Status::Ok();
    }

    // True when any replacement in `rewrite` binds `value` as a replaces
    // output target (EmitRewrite maps exactly those values into the
    // committed graph).
    const auto rewrite_binds_value = [](const RewriteEntry& rewrite, GraphValueId value) {
        return std::ranges::any_of(
                rewrite.replacements,
                [value](const ReplacementNode& replacement) {
                    return std::ranges::any_of(
                            replacement.outputs,
                            [value](const RewriteOutputBinding& output) {
                                return output.replaces.has_value() &&
                                       *output.replaces == value;
                            });
                });
    };
    // True when a replacement at index < prefix in `rewrite` binds `value`.
    const auto rewrite_prefix_binds_value = [](const RewriteEntry& rewrite,
                                               std::size_t prefix,
                                               GraphValueId value) {
        return std::ranges::any_of(
                rewrite.replacements.begin(),
                rewrite.replacements.begin() + static_cast<std::ptrdiff_t>(prefix),
                [value](const ReplacementNode& replacement) {
                    return std::ranges::any_of(
                            replacement.outputs,
                            [value](const RewriteOutputBinding& output) {
                                return output.replaces.has_value() &&
                                       *output.replaces == value;
                            });
                });
    };

    // Source topological positions; emission points are min positions over a
    // rewrite's old_nodes. A cyclic source graph is a graph-level error, not a
    // session invariant, so propagate it rather than asserting.
    AM_ASSIGN_OR_RETURN(const std::vector<GraphNodeId> order, graph_.TopologicalOrder());
    std::vector<std::size_t> position(graph_.GetNodes().size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        position[order[i].index] = i;
    }

    std::vector<std::size_t> emission_point(rewrites_.size());
    for (std::size_t i = 0; i < rewrites_.size(); ++i) {
        std::size_t earliest = std::numeric_limits<std::size_t>::max();
        for (const GraphNodeId old_node: rewrites_[i].old_nodes) {
            earliest = std::min(earliest, position[old_node.index]);
        }
        emission_point[i] = earliest;
    }

    for (std::size_t i = 0; i < rewrites_.size(); ++i) {
        const RewriteEntry& rewrite = rewrites_[i];
        if (!rewrite.active) {
            continue;
        }

        for (std::size_t r = 0; r < rewrite.replacements.size(); ++r) {
            const auto& replacement = rewrite.replacements[r];
            const std::string debug_ctx =
                    "replacement[" + std::to_string(r) + "]" +
                    (replacement.name.empty() ? "" : " '" + replacement.name + "'");

            for (const GraphValueId input: replacement.inputs) {
                const GraphValueId resolved = GetResolvedValue(input);
                if (resolved.index >= graph_.GetValues().size()) {
                    // Session constant or virtual value: always available.
                    continue;
                }

                const std::optional<GraphNodeId> producer =
                        graph_.GetValue(resolved).producer;
                if (!producer.has_value()) {
                    // Input/weight/constant/state: pre-mapped by
                    // CopyExternalValues before emission starts.
                    continue;
                }

                const auto rewrite_index = node_to_rewrite_[producer->index];
                const bool producer_rewritten =
                        rewrite_index.has_value() && *rewrite_index < rewrites_.size() &&
                        rewrites_[*rewrite_index].active;

                if (!producer_rewritten) {
                    // p untouched: live node, emitted at its own position.
                    if (position[producer->index] >= emission_point[i]) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession: rewrite " + std::to_string(i) +
                                " " + debug_ctx + " input value " +
                                std::to_string(resolved.index) + " is produced by node " +
                                std::to_string(producer->index) +
                                " which is not emitted before this rewrite's commit "
                                "emission point");
                    }
                    continue;
                }

                const RewriteEntry& producer_rewrite = rewrites_[*rewrite_index];
                if (!rewrite_binds_value(producer_rewrite, resolved)) {
                    // The covering rewrite never emits this value (e.g. a
                    // RemoveNode rewrite has no replacements), so Commit can
                    // never map the input; reject regardless of emission
                    // points.
                    return Status::InvalidArgument(
                            "GraphRewriteSession: rewrite " + std::to_string(i) + " " +
                            debug_ctx + " input value " + std::to_string(resolved.index) +
                            " is never produced during commit: its producer node " +
                            std::to_string(producer->index) + " is covered by rewrite " +
                            std::to_string(*rewrite_index) + " which does not bind this value");
                }

                if (*rewrite_index == i) {
                    // Producer inside this rewrite: the input is available
                    // only when an earlier replacement in the same group
                    // binds the value (EmitRewrite emits replacements in
                    // array order), matching Commit behavior.
                    if (!rewrite_prefix_binds_value(producer_rewrite, r, resolved)) {
                        return Status::InvalidArgument(
                                "GraphRewriteSession: rewrite " + std::to_string(i) + " " +
                                debug_ctx + " input value " + std::to_string(resolved.index) +
                                " is produced within the same rewrite but not by an "
                                "earlier replacement, so it cannot be mapped during commit");
                    }
                    continue;
                }

                if (emission_point[*rewrite_index] >= emission_point[i]) {
                    return Status::InvalidArgument(
                            "GraphRewriteSession: rewrite " + std::to_string(i) + " " +
                            debug_ctx + " input value " + std::to_string(resolved.index) +
                            " is produced by rewrite " + std::to_string(*rewrite_index) +
                            " which is not emitted before this rewrite's commit "
                            "emission point");
                }
            }
        }
    }
    return Status::Ok();
}

// Marks a rewrite as inactive; clears node_to_rewrite_ entries.
//
// Steps:
//   1. No-op (idempotent) when the rewrite is already inactive or out of
//      range.
//   2. Set active = false.
//   3. Reset the node_to_rewrite_ entry of each old_node that still points at
//      this rewrite.
void GraphRewriteSession::DeactivateRewrite(std::size_t rewrite_index) {
    // Idempotent: no-op if the rewrite is already inactive or out of range.
    if (rewrite_index >= rewrites_.size() || !rewrites_[rewrite_index].active) {
        return;
    }

    rewrites_[rewrite_index].active = false;
    for (const auto old_node: rewrites_[rewrite_index].old_nodes) {
        if (old_node.index < node_to_rewrite_.size() &&
            node_to_rewrite_[old_node.index] == rewrite_index) {
            node_to_rewrite_[old_node.index].reset();
        }
    }
}

// Marks the consumer index stale after any structural mutation.
//
// Steps:
//   1. Bump mutation_generation_.
//   2. Drop the cached consumer index so the next touch rebuilds it.
void GraphRewriteSession::InvalidateConsumerCache() noexcept {
    ++mutation_generation_;
    consumer_cache_.reset();
}

StatusOr<GraphValueId> SubgraphBuilder::Emit(
        OpType op_type,
        std::vector<GraphValueId> inputs,
        NodeOutputDesc output_desc,
        OpParams op_params,
        std::optional<uint32_t> decoder_layer_index,
        std::string name) {
    std::vector<NodeOutputDesc> output_descs;
    output_descs.push_back(std::move(output_desc));
    AM_ASSIGN_OR_RETURN(std::vector<GraphValueId> outputs,
                        Emit(op_type,
                             std::move(inputs),
                             std::move(output_descs),
                             std::move(op_params),
                             decoder_layer_index,
                             std::move(name)));
    AM_CHECK(outputs.size() == 1, "SubgraphBuilder::Emit single-output wrapper expected one output");
    return outputs[0];
}

StatusOr<std::vector<GraphValueId>> SubgraphBuilder::Emit(
        OpType op_type,
        std::vector<GraphValueId> inputs,
        std::vector<NodeOutputDesc> output_descs,
        OpParams op_params,
        std::optional<uint32_t> decoder_layer_index,
        std::string name) {
    // Normalize inputs: a virtual id that has already been Yielded to a
    // source value is replaced by that source value, so consumers emitted
    // after the Yield still reference the value the subgraph produces.
    for (auto& input: inputs) {
        if (const auto it = aliases_.find(input); it != aliases_.end()) {
            input = it->second;
        }
    }

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
            .name = std::move(name),
    };

    node.outputs.reserve(output_descs.size());
    for (auto& desc: output_descs) {
        const auto id = session_.AllocateVirtualValue();
        virtual_ids.push_back(id);
        node.outputs.push_back({
                .desc = std::move(desc),
                .replaces = id,
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

    // Reject re-yield: the binding was already redirected, so honoring a
    // second Yield would silently drop the original target (and the lookup
    // below would misreport the value as never emitted).
    if (const auto it = aliases_.find(internal_val); it != aliases_.end()) {
        return Status::InvalidArgument(
                "SubgraphBuilder::Yield: internal value " +
                std::to_string(internal_val.index) +
                " has already been yielded to value " +
                std::to_string(it->second.index));
    }

    for (auto& node: new_nodes_) {
        for (auto& out: node.outputs) {
            if (out.replaces == internal_val) {
                // Record the alias and redirect every pending consumer that
                // still references the virtual id to the replacement target,
                // so the internal edge survives the binding change.
                aliases_.emplace(internal_val, old_value_to_replace);
                NormalizeInputs(internal_val, old_value_to_replace);
                out.replaces = old_value_to_replace;
                return Status::Ok();
            }
        }
    }
    return Status::InvalidArgument(
            "SubgraphBuilder::Yield: internal_val was not produced by any Emit call");
}

// Redirects every pending replacement input referencing `from` to `to`.
//
// Steps:
//   1. Walk all accumulated replacement nodes.
//   2. Replace each input equal to `from` with `to`.
void SubgraphBuilder::NormalizeInputs(GraphValueId from, GraphValueId to) noexcept {
    for (auto& node: new_nodes_) {
        for (auto& input: node.inputs) {
            if (input == from) {
                input = to;
            }
        }
    }
}

Status SubgraphBuilder::Commit() {
    // Final normalization pass (defense in depth): Emit already normalizes
    // new inputs and Yield rewrites existing ones; this covers any input
    // that still references a yielded virtual id.
    for (const auto& [virtual_id, source_id]: aliases_) {
        NormalizeInputs(virtual_id, source_id);
    }

    // Submit accumulated replacement nodes; on success, clear internal state
    // so the builder can be reused for another Emit/Yield/Commit cycle.
    Status status = session_.ReplaceSubgraph(old_nodes_, new_nodes_);
    if (status.ok()) {
        new_nodes_.clear();
        aliases_.clear();
    }
    return status;
}

}// namespace aethermind
