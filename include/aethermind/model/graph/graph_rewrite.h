#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H

/// @file graph_rewrite.h
/// @brief Rewrite session and builder for transforming ModelGraph instances.
///
/// GraphRewriteSession records graph mutations (RemoveNode, ReplaceSubgraph,
/// RedirectInput, ReplaceValue) as deltas over an immutable source ModelGraph,
/// while providing query methods that reflect both the original graph and
/// pending changes. Commit() materializes the result into a new owned ModelGraph.
///
/// The session manages three value id spaces:
///   - source values from the original graph (index < graph_.GetValues().size())
///   - session constants added via AddConstant (index >= source range)
///   - virtual values for internal edges within a SubgraphReplacement
///     (index >= source range, not a session constant)
///
/// @section Ownership
/// The session borrows a const ModelGraph& — the caller must keep the source
/// graph alive. Commit() produces a new owned ModelGraph.
///
/// @section Thread-safety
/// Not thread-safe. All methods are for single-threaded use within one pass.

#include "aethermind/model/graph/graph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace aethermind {

/// Removes a single node and all its output values from the session view.
struct NodeRemoval {
    GraphNodeId node{};
};

/// Rewires one input port of a live node to a different value.
struct InputRedirection {
    GraphNodeId node{};
    size_t input_index = 0;
    GraphValueId new_value{};
};

/// Redirects all consumers of `old_value` to `new_value` after resolution.
struct ValueReplacement {
    GraphValueId old_value{};
    GraphValueId new_value{};
};

/// Binds one output of a replacement node to a value identity.
///
/// When `replaces` targets a source graph value, that value's identity is
/// taken over by this output (consumers see it as the replacement). When it
/// targets a virtual value, the output serves as an internal edge within a
/// multi-node subgraph replacement.
struct RewriteOutputBinding {
    NodeOutputDesc desc{};
    std::optional<GraphValueId> replaces{};
};

/// Describes a node to be inserted by ReplaceSubgraph.
///
/// Inputs may reference source values, session constants, or virtual values
/// allocated within the same replacement group. Each output specifies which
/// existing value it replaces (see RewriteOutputBinding).
struct ReplacementNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<RewriteOutputBinding> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};
    std::string debug_name{};
};

/// Replaces a set of source graph nodes with new replacement nodes.
///
/// An empty `replacement_nodes` vector acts as a removal of the old nodes.
struct SubgraphReplacement {
    std::vector<GraphNodeId> old_nodes{};
    std::vector<ReplacementNode> replacement_nodes{};
};

/// Variant over all mutation types for batch submission via Apply().
using GraphMutation = std::variant<SubgraphReplacement,
                                   NodeRemoval,
                                   InputRedirection,
                                   ValueReplacement>;

/// Snapshot of a live node's current state in the session.
///
/// Inputs are resolved through GetResolvedValue (reflects RedirectInput and
/// ReplaceValue). Outputs are the original graph value ids — callers that
/// need the terminal value after replacement should call GetResolvedValue
/// on each output id. The view is valid only while the session is alive.
struct GraphNodeView {
    GraphNodeId node{};
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<GraphValueId> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};
    std::string debug_name{};
};

/// Records pending rewrites over an immutable source ModelGraph and
/// materializes the result on Commit().
///
/// The session tracks three categories of value ids:
///   - source values: original graph values, index < graph_.GetValues().size()
///   - session constants: added via AddConstant(), index >= source range
///   - virtual values: internal edges within a subgraph replacement,
///     index >= source range, not a session constant
///
/// Node liveness (IsNodeLive): a node is live when untouched or when only
/// modified via RedirectInput (which installs a mirror replacement). Nodes
/// removed via RemoveNode or replaced via ReplaceSubgraph are not live.
///
/// Value liveness (IsValueLive): a value is structurally present when its
/// producer is live or an active subgraph replacement takes over its identity.
/// ReplaceValue does not affect liveness — only consumer resolution.
///
/// @section Thread-safety
/// Not thread-safe. All methods are for single-threaded pass use. Const methods
/// use mutable caching internally and must not be called concurrently.
class GraphRewriteSession {
public:
    /// Constructs a session over a source graph. The graph must outlive the session.
    explicit GraphRewriteSession(const ModelGraph& graph);

    /// Allocates a virtual value id for internal use within a subgraph replacement.
    /// Virtual values serve as edges between replacement nodes and are not
    /// persisted in the committed graph.
    AM_NODISCARD GraphValueId AllocateVirtualValue();

    /// Adds a new constant value scoped to this session.
    /// The constant is materialized in the committed graph during Commit().
    /// Returns a session value id (index >= source graph values).
    AM_NODISCARD GraphValueId AddConstant(TensorSpec spec,
                                          ConstantBinding binding,
                                          QuantizationSpec quantization,
                                          std::string debug_name);

    /// Applies a batch of mutations sequentially. Non-atomic: if mutation N
    /// fails, mutations 0..N-1 remain applied to the session. The caller is
    /// responsible for either committing the partial state or discarding the
    /// session.
    AM_NODISCARD Status Apply(std::span<const GraphMutation> mutations);
    /// Removes a live node. Equivalent to ReplaceSubgraph({node}, {}).
    /// The node must be live per IsNodeLive.
    AM_NODISCARD Status RemoveNode(GraphNodeId node);

    /// Replaces a set of source nodes with replacement nodes.
    ///
    /// At least one old_node is required. Replacement nodes are emitted in the
    /// order they appear in the vector when the first old_node is encountered
    /// in topological order during Commit(). Their inputs must resolve to values
    /// that are available at that point (external values, already-emitted live
    /// nodes, session constants, or virtual values produced by earlier
    /// replacements within the same rewrite).
    ///
    /// Deactivates any existing rewrite that covers any of the old_nodes.
    /// An empty replacement_nodes vector acts as removal.
    AM_NODISCARD Status ReplaceSubgraph(std::span<const GraphNodeId> old_nodes,
                                        const std::vector<ReplacementNode>& replacement_nodes);

    /// Rewires one input of a live node to a different value.
    ///
    /// The new_value must be a source value or a session constant; virtual
    /// values are not permitted (they are rewrite-internal and have no
    /// committed-graph identity).
    ///
    /// Untouched nodes are represented by installing a mirror replacement.
    /// Nodes that already have a mirror rewrite accumulate the new input
    /// change. Returns InvalidArgument for nodes that are not live.
    AM_NODISCARD Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);

    /// Redirects consumers of `old_value` to `new_value` after resolution.
    ///
    /// The old_value must be a source graph value. The new_value may be a source
    /// value or a session constant. Virtual values are not permitted.
    /// Detects and rejects replacement cycles.
    AM_NODISCARD Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);

    /// Walks the replacement chain from `value` to its terminal.
    ///
    /// Uses a mutable cache with path compression: all values along the
    /// resolution path are cached to the terminal value for O(1) subsequent
    /// lookups. Out-of-range ids return identity.
    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;

    /// Returns a snapshot of a live node with inputs resolved through
    /// GetResolvedValue. Outputs are the original graph value ids.
    /// Returns NotFound if the node is not live.
    AM_NODISCARD StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;

    /// Returns true if `node` is currently observable in the session.
    ///
    /// A node is live when no rewrite has touched it, or when it has been
    /// modified only via RedirectInput (which installs a mirror replacement
    /// that still exposes the original node identity). Nodes removed via
    /// RemoveNode or replaced via ReplaceSubgraph are not live.
    ///
    /// Out-of-range ids return false. This method is the single source of
    /// truth for node liveness; GetNodeView and the enumeration APIs below
    /// are defined in terms of it.
    AM_NODISCARD bool IsNodeLive(GraphNodeId node) const noexcept;

    /// Returns live node ids in topological order, following the original
    /// graph's ordering. Filters out nodes that have been removed or replaced
    /// in this session.
    ///
    /// Returns the underlying ModelGraph::TopologicalOrder error if the
    /// original graph contains a cycle (the session does not introduce new
    /// edges, so a cycle can only originate from the source graph).
    AM_NODISCARD StatusOr<std::vector<GraphNodeId>> GetTopologicalOrder() const;

    /// Returns live node ids whose op_type matches `op_type`, in ascending
    /// node-index order. Filters out nodes that have been removed or replaced
    /// in this session.
    ///
    /// Note: this reflects the op_type of the *original* graph node, not the
    /// op_type of any RedirectInput mirror replacement. RedirectInput only
    /// changes input wiring, not the operator itself.
    AM_NODISCARD std::vector<GraphNodeId> FindNodesByOpType(OpType op_type) const;

    /// Returns true if `value` is structurally present in the current session view.
    ///
    /// This is analogous to IsNodeLive(): it answers whether the value still exists
    /// after session-local structural rewrites. It does not mean the value is
    /// reachable from graph outputs or required by DCE. A structurally present value
    /// may still be dead-code-elimination eligible if it has no live consumers and
    /// is not a graph output / side effect root.
    ///
    /// A value is live when any of the following holds:
    /// - It is an external value (input, weight, constant, state) with no producer.
    /// - Its producer is a live node (e.g., untouched or only RedirectInput'd).
    /// - An active rewrite's replacement output takes over the value via its
    ///   `replaces` binding, even if the original producer was removed/replaced.
    ///
    /// ReplaceValue does not affect liveness: a replaced value still exists,
    /// only its consumers are redirected.
    AM_NODISCARD bool IsValueLive(GraphValueId value) const noexcept;

    /// Returns true if `value` resolves to a compile-time constant.
    ///
    /// Resolves through any ReplaceValue chain first, then checks both source
    /// graph constants (ConstantValue payload) and session constants added
    /// via AddConstant. Out-of-range and virtual values return false.
    AM_NODISCARD bool IsConstant(GraphValueId value) const;

    /// Returns true if all resolved inputs of `node` are compile-time constants.
    ///
    /// Uses GetNodeView internally so inputs reflect RedirectInput and
    /// ReplaceValue resolution. Returns false if the node is not live.
    /// Nodes with no inputs return true (vacuous).
    AM_NODISCARD bool AreAllInputsConstant(GraphNodeId node) const;

    /// Returns the output descriptor represented by a real graph value.
    ///
    /// This is derived from the source graph value's spec, payload,
    /// quantization, and debug name. Virtual values and out-of-range values
    /// return InvalidArgument.
    AM_NODISCARD StatusOr<NodeOutputDesc> GetValueOutputDesc(GraphValueId value) const;

    /// Returns true when `value` is directly marked as a graph output in the
    /// source graph. Out-of-range ids and virtual values return false.
    AM_NODISCARD bool IsGraphOutput(GraphValueId value) const noexcept;

    /// Returns all live value ids in ascending index order. Excludes values
    /// produced by removed/replaced nodes that no replacement takes over, and
    /// virtual values (they are rewrite-internal).
    AM_NODISCARD std::vector<GraphValueId> GetLiveValues() const;

    /// Returns live original graph nodes that consume `value` (after resolution)
    /// as an input, in topological order. A node appears at most once even if
    /// it consumes the value on multiple input ports.
    ///
    /// Both `value` and each node's inputs are resolved via GetResolvedValue
    /// before comparison, so ReplaceValue is accounted for: querying consumers
    /// of a replaced value returns the same result as querying consumers of
    /// its resolution target.
    ///
    /// Note: this only returns live ORIGINAL graph nodes (untouched or
    /// RedirectInput'd). Active replacement nodes from ReplaceSubgraph are NOT
    /// included, because they don't have GraphNodeIds in the session's
    /// original-graph id space. For DCE liveness checking, use
    /// HasLiveConsumers() which accounts for both original and replacement
    /// consumers.
    ///
    /// Virtual values and out-of-range ids return an empty vector.
    AM_NODISCARD std::vector<GraphNodeId> FindConsumers(GraphValueId value) const;

    /// Returns true if any live node or active replacement node consumes
    /// `value` (after resolution) as an input.
    ///
    /// This is the correct consumer check for DCE: for a structurally live
    /// value, DCE may treat it as dead only if HasLiveConsumers(v) is false AND
    /// v is not a graph output. FindConsumers() alone is insufficient because
    /// it excludes replacement node consumers.
    ///
    /// Both `value` and each consumer's inputs (original node inputs via
    /// GetNodeView, replacement node inputs directly) are resolved via
    /// GetResolvedValue before comparison, so ReplaceValue is accounted for.
    ///
    /// Virtual values and out-of-range ids return false.
    AM_NODISCARD bool HasLiveConsumers(GraphValueId value) const;

    /// Validates the session's internal consistency without materializing.
    ///
    /// Checks: replacement targets are valid, old_node ids are in range,
    /// replacement node inputs/outputs are valid, virtual values satisfy
    /// ordering constraints within and across rewrites.
    AM_NODISCARD Status ValidateEdits() const;

    /// Materializes the session state into a new ModelGraph.
    ///
    /// Steps: CopyExternalValues -> topological traversal (emitting rewrites
    /// and surviving original nodes) -> MarkCommittedOutputs -> Validate.
    /// Returns InvalidArgument if the result would be invalid.
    AM_NODISCARD StatusOr<ModelGraph> Commit() const;

    /// Validates that `value` refers to a source graph value or a session-added
    /// constant. Returns InvalidArgument if the id is out of range or refers to
    /// a virtual value allocated by AllocateVirtualValue().
    AM_NODISCARD Status CheckValueId(GraphValueId value) const;

private:
    friend class SubgraphBuilder;

    /// Records one rewrite operation within a session.
    /// Active rewrites are emitted during Commit; deactivated ones are
    /// overridden by a subsequent overlapping mutation.
    struct RewriteEntry {
        std::vector<GraphNodeId> old_nodes{};
        std::vector<ReplacementNode> replacements{};
        bool active = true;
        // When true, IsNodeLive returns true for the sole old_node and
        // GetNodeView exposes a live node view (used by RedirectInput).
        bool exposes_node_view = false;
    };

    /// Metadata for a session-local constant added via AddConstant.
    struct SessionConstant {
        TensorSpec spec{};
        ConstantBinding binding{};
        QuantizationSpec quantization{};
        std::string debug_name{};
    };

    /// Temporary mapping tables used during Commit() to translate value ids
    /// from source/session spaces into the committed graph's id space.
    using ValueMap = std::vector<std::optional<GraphValueId>>;
    struct CommitValueMaps {
        ValueMap& source_values;    // Source value -> committed value
        ValueMap& session_constants;// Session constant -> committed constant
        ValueMap& virtual_values;   // Virtual value -> committed output
    };

    /// Cached consumer index for one mutation generation.
    /// Keys are resolved value ids, so ReplaceValue aliases share consumers.
    struct ConsumerCache {
        std::uint64_t generation = 0;
        std::vector<std::vector<GraphNodeId>> original_consumers{};
        std::vector<uint32_t> replacement_consumer_counts{};
    };

    // Returns InvalidArgument if node id is out of range.
    AM_NODISCARD Status CheckNodeId(GraphNodeId node) const;
    // Returns InvalidArgument if value is not a source graph value.
    AM_NODISCARD Status CheckSourceValueId(GraphValueId value) const;
    // Returns InvalidArgument for out-of-range ids; accepts virtual values.
    AM_NODISCARD Status CheckValueIdAllowVirtual(GraphValueId value) const;
    // True when value is in the session id space (index >= source range).
    AM_NODISCARD bool IsSessionValue(GraphValueId value) const noexcept;
    // True when value is a session constant (session value + has SessionConstant metadata).
    AM_NODISCARD bool IsSessionConstant(GraphValueId value) const noexcept;
    // True when value is a virtual value (session value, not a session constant).
    AM_NODISCARD bool IsVirtualValue(GraphValueId value) const noexcept;
    // True when any active rewrite's replacement output takes over this value.
    AM_NODISCARD bool IsValueReplacedByActiveRewrite(GraphValueId value) const noexcept;
    // Returns the 0-based index into the session value space.
    AM_NODISCARD std::size_t GetVirtualIndex(GraphValueId virtual_id) const noexcept {
        return virtual_id.index - graph_.GetValues().size();
    }
    // Validates that replacement node inputs/outputs reference valid ids.
    AM_NODISCARD Status ValidateReplacementNode(const ReplacementNode& replacement) const;
    // Validates that replacement targets belong to old_nodes and are not duplicated.
    AM_NODISCARD Status ValidateReplacementTargets(
            std::span<const GraphNodeId> old_nodes,
            const std::vector<ReplacementNode>& replacement_nodes) const;
    // Validates virtual value ordering (no consumption before production).
    AM_NODISCARD Status ValidateVirtualValues() const;
    // Builds a NodeOutputDesc from a session constant's metadata.
    AM_NODISCARD NodeOutputDesc MakeOutputDescFromSessionConstant(GraphValueId value) const;
    // Translates a value id from source/session space to committed graph space.
    AM_NODISCARD StatusOr<GraphValueId> MapCommittedValue(
            GraphValueId value,
            const CommitValueMaps& maps) const;
    // Copies source external values and session constants into committed graph.
    AM_NODISCARD Status CopyExternalValues(ModelGraph& committed,
                                           CommitValueMaps& maps) const;
    // Emits all replacement nodes in a rewrite entry into the committed graph.
    AM_NODISCARD Status EmitRewrite(const RewriteEntry& rewrite,
                                    ModelGraph& committed,
                                    CommitValueMaps& maps) const;
    // Emits a single surviving original node into the committed graph.
    AM_NODISCARD Status EmitOriginalNode(GraphNodeId old_node,
                                         ModelGraph& committed,
                                         CommitValueMaps& maps) const;
    // Maps graph output values through resolution into committed graph.
    AM_NODISCARD Status MarkCommittedOutputs(ModelGraph& committed,
                                             const CommitValueMaps& maps) const;
    // Marks a rewrite as inactive; clears node_to_rewrite_ entries.
    void DeactivateRewrite(std::size_t rewrite_index);
    // Invalidates cached consumer indexes after a successful state mutation.
    void InvalidateConsumerCache() noexcept;
    // Builds or returns the consumer index for the current mutation generation.
    AM_NODISCARD const ConsumerCache& EnsureConsumerCache() const;

    // Immutable source graph; must outlive the session.
    const ModelGraph& graph_;
    // Total count of allocated session values (constants + virtual).
    std::size_t virtual_value_count_ = 0;
    // Per-session-value metadata: has_value = session constant, nullopt = virtual.
    // Index i corresponds to session value at graph_.GetValues().size() + i.
    std::vector<std::optional<SessionConstant>> session_constants_{};
    // Active and deactivated rewrites, in submission order.
    std::vector<RewriteEntry> rewrites_{};
    // Maps each source node to the rewrite that covers it, or nullopt.
    // Sized to graph_.GetNodes().size() at construction.
    std::vector<std::optional<std::size_t>> node_to_rewrite_{};
    // Maps each source value to its replacement target, or nullopt.
    // Sized to graph_.GetValues().size() at construction.
    std::vector<std::optional<GraphValueId>> value_replacements_{};
    // Incremented after successful mutations that can change consumer queries.
    std::uint64_t mutation_generation_ = 0;
    // Caches GetResolvedValue results for O(1) subsequent lookups.
    // Invalidated on each ReplaceValue. Mutable for logical const.
    mutable std::vector<std::optional<GraphValueId>> resolved_value_cache_{};
    // Lazily built consumer index for FindConsumers/HasLiveConsumers.
    mutable std::optional<ConsumerCache> consumer_cache_{};
};

/// Convenience builder for constructing subgraph replacements without
/// manually managing virtual value allocation, RewriteOutputBinding wiring,
/// and ReplacementNode assembly.
///
/// Emit allocates virtual values for internal edges and returns their ids.
/// Yield redirects a virtual value (returned by Emit) to replace a real
/// source graph value. Commit submits the accumulated replacement nodes
/// to the session as a single ReplaceSubgraph mutation.
///
/// Usage:
///   SubgraphBuilder builder(session, {old_node1, old_node2});
///   GraphValueId mid = builder.Emit(OpType::kSilu, {input}, output_desc, params);
///   GraphValueId out = builder.Emit(OpType::kMul, {mid, other}, output_desc, params);
///   builder.Yield(out, old_output_value);
///   builder.Commit();
///
/// The builder is reusable after Commit() clears its internal state.
///
/// Thread-safety: not thread-safe. Each builder instance is intended for
/// single-threaded use within one pass invocation.
class SubgraphBuilder {
public:
    SubgraphBuilder(GraphRewriteSession& session, std::vector<GraphNodeId> old_nodes)
        : session_(session), old_nodes_(std::move(old_nodes)) {}

    /// Creates a new replacement node with a single output described by
    /// `output_desc`. Allocates a virtual value internally, binds it as the
    /// node's output, and returns the virtual value id for use as input to
    /// subsequent Emit calls or as the internal_val argument to Yield.
    GraphValueId Emit(OpType op_type,
                      std::vector<GraphValueId> inputs,
                      NodeOutputDesc output_desc,
                      OpParams op_params = std::monostate{},
                      std::optional<uint32_t> decoder_layer_index = std::nullopt,
                      std::string debug_name = {});

    /// Creates a new replacement node with one output per descriptor in
    /// `output_descs`. Allocates one virtual value per output and returns the
    /// virtual values in descriptor order.
    std::vector<GraphValueId> Emit(OpType op_type,
                                   std::vector<GraphValueId> inputs,
                                   std::vector<NodeOutputDesc> output_descs,
                                   OpParams op_params = std::monostate{},
                                   std::optional<uint32_t> decoder_layer_index = std::nullopt,
                                   std::string debug_name = {});

    /// Marks an internal virtual value (returned by Emit) as the replacement
    /// for an external graph value. After Commit, all consumers of
    /// old_value_to_replace will consume the new node's output instead.
    ///
    /// Returns InvalidArgument if internal_val was not produced by any
    /// prior Emit call.
    Status Yield(GraphValueId internal_val, GraphValueId old_value_to_replace);

    /// Submits the accumulated replacement nodes to the session as a single
    /// ReplaceSubgraph mutation. On success, the builder is reset and can be
    /// reused for further Emit/Yield/Commit cycles.
    Status Commit();

private:
    GraphRewriteSession& session_;
    std::vector<GraphNodeId> old_nodes_;
    std::vector<ReplacementNode> new_nodes_;
};

}// namespace aethermind

#endif
