#ifndef AETHERMIND_GRAPH_OPTIMIZATION_GRAPH_REWRITE_H
#define AETHERMIND_GRAPH_OPTIMIZATION_GRAPH_REWRITE_H

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
///   - session constants added via AddSessionConstant (index >= source range)
///   - virtual values for internal edges within a SubgraphReplacement
///     (index >= source range, not a session constant)
///
/// @section Ownership
/// The session borrows a const ModelGraph& — the caller must keep the source
/// graph alive. Commit() produces a new owned ModelGraph.
///
/// @section Thread-safety
/// Not thread-safe. All methods are for single-threaded use within one pass.

#include "aethermind/graph/graph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace aethermind {

/// @brief Removes a single node and all its output values from the session view.
struct NodeRemoval {
    GraphNodeId node{};
};

/// @brief Rewires one input port of a live node to a different value.
struct InputRedirection {
    GraphNodeId node{};
    size_t input_index = 0;
    GraphValueId new_value{};
};

/// @brief Redirects all consumers of `old_value` to `new_value` after resolution.
struct ValueReplacement {
    GraphValueId old_value{};
    GraphValueId new_value{};
};

/// @brief Binds one output of a replacement node to a value identity.
///
/// When `replaces` targets a source graph value, that value's identity is
/// taken over by this output (consumers see it as the replacement). When it
/// targets a virtual value, the output serves as an internal edge within a
/// multi-node subgraph replacement.
struct RewriteOutputBinding {
    NodeOutputDesc desc{};
    std::optional<GraphValueId> replaces{};
};

/// @brief Describes a node to be inserted by ReplaceSubgraph.
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
    std::string name{};
};

/// @brief Replaces a set of source graph nodes with new replacement nodes.
///
/// An empty `replacement_nodes` vector acts as a removal of the old nodes.
struct SubgraphReplacement {
    std::vector<GraphNodeId> old_nodes{};
    std::vector<ReplacementNode> replacement_nodes{};
};

/// @brief Variant over all mutation types for batch submission via Apply().
using GraphMutation = std::variant<SubgraphReplacement,
                                   NodeRemoval,
                                   InputRedirection,
                                   ValueReplacement>;

/// @brief Snapshot of a live node's current state in the session.
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
    std::string name{};
};

/// @brief Controls optional behavior when materializing a rewrite session.
///
/// Commit-time pruning removes unreachable residue from the mixed
/// source/replacement graph: replacement nodes and surviving source nodes
/// that no path from a graph output (after resolution) can reach are
/// omitted from the committed graph. Pruning runs either on an explicit
/// request (see RequestCommitPruning) or when `force_prune_unreachable` is
/// set. DCE and Commit pruning are complementary: DeadCodeEliminationPass
/// deletes dead source nodes from the session state during the run, while
/// Commit pruning is the fallback that drops unreachable residual rewrites
/// and their source producers when the whole graph is materialized.
struct CommitOptions {
    /// Force pruning even without a session request. Additive only: passing
    /// false cannot disable a RequestCommitPruning() request.
    bool force_prune_unreachable = false;
};

/// @brief Records pending rewrites over an immutable source ModelGraph and
/// materializes the result on Commit().
///
/// The session tracks three categories of value ids:
///   - source values: original graph values, index < graph_.GetValues().size()
///   - session constants: added via AddSessionConstant(), index >= source range
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
    /// @brief Constructs a session over a source graph.
    /// @param graph Source graph to rewrite; must outlive the session.
    explicit GraphRewriteSession(const ModelGraph& graph);

    /// @brief Allocates a virtual value id for internal use within a subgraph replacement.
    ///
    /// Virtual values serve as edges between replacement nodes and are not
    /// persisted in the committed graph.
    /// @return Newly allocated virtual value id.
    AM_NODISCARD GraphValueId AllocateVirtualValue();

    /// @brief Adds a new session constant value scoped to this session.
    ///
    /// The constant is allocated in the session-local id space
    /// (ValueKind::kSessionConstant). Commit() materializes it only when the
    /// final committed graph references it through a retained consumer — a
    /// live source node or a retained replacement after pruning — or a graph
    /// output resolved by ReplaceValue; otherwise it is discarded (see
    /// CopyExternalValues for the lockstep pruning alignment).
    /// @param spec Tensor spec for the constant value.
    /// @param binding Constant binding payload backing the value.
    /// @param quantization Quantization metadata for the constant.
    /// @param name Debug name used as the AddSessionConstant tag.
    /// @return Session-local constant value id (index >= source graph values).
    AM_NODISCARD GraphValueId AddSessionConstant(TensorSpec spec,
                                                 ConstantBinding binding,
                                                 QuantizationSpec quantization,
                                                 std::string name);

    /// @brief Applies a batch of mutations sequentially.
    ///
    /// Non-atomic: if mutation N fails, mutations 0..N-1 remain applied to the
    /// session. The caller is responsible for either committing the partial
    /// state or discarding the session.
    /// @param mutations Mutations to apply, in order.
    /// @return Status::Ok() on success, or the first error encountered.
    Status Apply(std::span<const GraphMutation> mutations);
    /// @brief Removes a live node. Equivalent to ReplaceSubgraph({node}, {}).
    /// @param node Node to remove; must be live per IsNodeLive.
    /// @return Status::Ok() on success, or the first error encountered.
    Status RemoveNode(GraphNodeId node);

    /// @brief Replaces a set of source nodes with replacement nodes.
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
    /// @param old_nodes Source graph nodes to replace; at least one is required.
    /// @param replacement_nodes New nodes to emit; empty acts as removal.
    /// @return Status::Ok() on success, or the first error encountered.
    Status ReplaceSubgraph(std::span<const GraphNodeId> old_nodes,
                           const std::vector<ReplacementNode>& replacement_nodes);

    /// @brief Rewires one input of a live node to a different value.
    ///
    /// The new_value must be a source value or a session constant; virtual
    /// values are not permitted (they are rewrite-internal and have no
    /// committed-graph identity).
    ///
    /// Untouched nodes are represented by installing a mirror replacement.
    /// Nodes that already have a mirror rewrite accumulate the new input
    /// change. Returns InvalidArgument for nodes that are not live.
    /// @param node Live node whose input is rewired.
    /// @param input_index 0-based input port index to rewire.
    /// @param new_value Replacement value; must be a source value or session constant.
    /// @return Status::Ok() on success, or InvalidArgument if the node is not live.
    Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);

    /// @brief Redirects consumers of `old_value` to `new_value` after resolution.
    ///
    /// The old_value must be a source graph value. The new_value may be a source
    /// value or a session constant. Virtual values are not permitted.
    /// Detects and rejects replacement cycles. The installation itself does not
    /// compare specs; ValidateEdits/Commit reject replacements whose resolved
    /// terminal changes the replaced value's dtype or shape identity.
    /// @param old_value Source graph value whose consumers are redirected.
    /// @param new_value Resolution target; source value or session constant.
    /// @return Status::Ok() on success, or the first error encountered.
    Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);

    /// @brief Walks the replacement chain from `value` to its terminal.
    ///
    /// Uses a mutable cache with path compression: all values along the
    /// resolution path are cached to the terminal value for O(1) subsequent
    /// lookups. Out-of-range ids return identity.
    /// @param value Value id to resolve.
    /// @return Terminal value id at the end of the replacement chain.
    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;

    /// @brief Returns a snapshot of a live node with inputs resolved through
    /// GetResolvedValue. Outputs are the original graph value ids.
    /// @param node Node id to snapshot.
    /// @return GraphNodeView reflecting resolved inputs, or NotFound if not live.
    StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;

    /// @brief Returns true if `node` is currently observable in the session.
    ///
    /// A node is live when no rewrite has touched it, or when it has been
    /// modified only via RedirectInput (which installs a mirror replacement
    /// that still exposes the original node identity). Nodes removed via
    /// RemoveNode or replaced via ReplaceSubgraph are not live.
    ///
    /// Out-of-range ids return false. This method is the single source of
    /// truth for node liveness; GetNodeView and the enumeration APIs below
    /// are defined in terms of it.
    /// @param node Node id to test.
    /// @return True if the node is live; false for removed/replaced or out-of-range ids.
    AM_NODISCARD bool IsNodeLive(GraphNodeId node) const noexcept;

    /// @brief Returns live node ids in the source graph's topological order,
    /// filtered by liveness. Filters out nodes that have been removed or
    /// replaced in this session.
    ///
    /// @return Live node ids in source-graph topological order, or the
    ///         underlying ModelGraph::TopologicalOrder error if the original
    ///         graph contains a cycle. RemoveNode and ReplaceSubgraph do not
    ///         introduce cross-node edges, so the source ordering remains valid
    ///         for those mutations. RedirectInput, however, can rewire a node's
    ///         input to a value produced by a later node, creating a back-edge
    ///         that violates this ordering; such cases are not detected here and
    ///         surface as a Commit failure ("producer not yet emitted").
    StatusOr<std::vector<GraphNodeId>> GetTopologicalOrder() const;

    /// @brief Returns live node ids whose op_type matches `op_type`, in ascending
    /// node-index order. Filters out nodes that have been removed or replaced
    /// in this session.
    ///
    /// Note: this reflects the op_type of the *original* graph node, not the
    /// op_type of any RedirectInput mirror replacement. RedirectInput only
    /// changes input wiring, not the operator itself.
    /// @param op_type Operator type to match against original node op_types.
    /// @return Matching live node ids in ascending node-index order.
    AM_NODISCARD std::vector<GraphNodeId> FindNodesByOpType(OpType op_type) const;

    /// @brief Returns true if `value` is structurally present in the current session view.
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
    /// @param value Value id to test.
    /// @return True if the value is structurally present; false otherwise.
    AM_NODISCARD bool IsValueLive(GraphValueId value) const noexcept;

    /// @brief Returns true if `value` resolves to a compile-time constant.
    ///
    /// Resolves through any ReplaceValue chain first, then checks both source
    /// graph constants (ConstantValue payload) and session constants added
    /// via AddSessionConstant. Out-of-range and virtual values return false.
    /// @param value Value id to test.
    /// @return True if the value resolves to a compile-time constant; false otherwise.
    AM_NODISCARD bool IsConstant(GraphValueId value) const;

    /// @brief Returns true if all resolved inputs of `node` are compile-time constants.
    ///
    /// Uses GetNodeView internally so inputs reflect RedirectInput and
    /// ReplaceValue resolution. Returns false if the node is not live.
    /// Nodes with no inputs return true (vacuous).
    /// @param node Node id whose inputs are checked.
    /// @return True if all resolved inputs are constants (vacuously true for no inputs);
    ///         false if the node is not live or any input is non-constant.
    AM_NODISCARD bool AreAllInputsConstant(GraphNodeId node) const;

    /// @brief Returns the spec-bearing descriptor for an existing graph value.
    ///
    /// This is derived from the source graph value's spec, payload,
    /// quantization, and debug name. Virtual values and out-of-range values
    /// return InvalidArgument.
    /// @param value Value id to describe.
    /// @return GraphValueDesc for the value, or InvalidArgument for virtual/out-of-range ids.
    StatusOr<GraphValueDesc> GetValueOutputMetadata(GraphValueId value) const;

    /// @brief Returns true when `value` is directly marked as a graph output in the
    /// source graph. Out-of-range ids and virtual values return false.
    /// @param value Value id to test.
    /// @return True if the value is a graph output; false for out-of-range/virtual ids.
    AM_NODISCARD bool IsGraphOutput(GraphValueId value) const noexcept;

    /// @brief Returns each graph output resolved to the terminal of its
    /// replacement chain (unreplaced outputs resolve to themselves).
    ///
    /// Commit marks exactly these terminals in the committed graph
    /// (MarkCommittedOutputs), so this is the authoritative set of values
    /// that must remain mapped at commit time. Passes that make liveness
    /// decisions (e.g. DeadCodeEliminationPass) must keep the producers of
    /// these terminals alive.
    /// @return Resolved terminal value id per source graph output, in output order.
    AM_NODISCARD std::vector<GraphValueId> GetResolvedGraphOutputs() const;

    /// @brief Returns all live value ids in ascending index order. Excludes values
    /// produced by removed/replaced nodes that no replacement takes over, and
    /// virtual values (they are rewrite-internal).
    /// @return Live value ids in ascending index order.
    AM_NODISCARD std::vector<GraphValueId> GetLiveValues() const;

    /// @brief Returns live original graph nodes that consume `value` (after resolution)
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
    /// @param value Value id whose consumers are queried.
    /// @return Consuming live original node ids in topological order (empty for
    ///         virtual/out-of-range ids), or the underlying
    ///         ModelGraph::TopologicalOrder error if the source graph contains
    ///         a cycle (the session does not introduce new edges, so a cycle
    ///         can only originate from the source graph).
    StatusOr<std::vector<GraphNodeId>> FindConsumers(GraphValueId value) const;

    /// @brief Returns true if any live node or active replacement node consumes
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
    /// @param value Value id whose consumers are checked.
    /// @return True if any live node or active replacement node consumes the value;
    ///         false for virtual/out-of-range ids or no consumers, or the
    ///         underlying ModelGraph::TopologicalOrder error if the source
    ///         graph contains a cycle (see FindConsumers).
    StatusOr<bool> HasLiveConsumers(GraphValueId value) const;

    /// @brief Validates the session's internal consistency without materializing.
    ///
    /// Checks: replacement targets are valid, value replacements preserve the
    /// replaced value's dtype and shape identity, old_node ids are in range,
    /// replacement node inputs/outputs are valid, virtual values satisfy
    /// ordering constraints within and across rewrites, and every active
    /// rewrite's replacement inputs are available at its commit emission point
    /// (source-value inputs must be produced by a node or rewrite that emits
    /// earlier, or be producer-less). When this returns Ok, Commit() will not
    /// fail on unmappable replacement inputs.
    /// @return Status::Ok() if consistent, or the first validation error.
    Status ValidateEdits() const;

    /// @brief Requests pruning of unreachable nodes at Commit time.
    ///
    /// This is a passive request, not a mode switch: it takes effect only on
    /// Commit() and only when the session actually has unreachable residue.
    /// DeadCodeEliminationPass issues it after its fixed-point loop has fully
    /// executed with its enabled flag set (an enable=false pass never issues
    /// the request). The request stays valid for the current session's
    /// lifetime and is never consumed by Commit(); repeated Commit() calls
    /// are stable. Checkpoint snapshots create a new session that does not
    /// inherit the request, so pruning never fires before DCE has run.
    void RequestCommitPruning() noexcept;

    /// @brief Materializes the session state into a new ModelGraph.
    ///
    /// By default the session is materialized faithfully: every active
    /// rewrite and every live source node is emitted. When the session
    /// carries a pruning request (see RequestCommitPruning) or
    /// `options.force_prune_unreachable` is set, Commit first computes the
    /// reverse-reachability closure of graph outputs through the mixed
    /// source/replacement graph, then emits only retained replacement and
    /// source nodes; unretained session constants are dropped in lockstep
    /// (see AddSessionConstant and CopyExternalValues).
    ///
    /// Steps: CopyExternalValues -> topological traversal (emitting rewrites
    /// and surviving original nodes, skipping unretained ones) ->
    /// MarkCommittedOutputs -> Validate.
    /// Runs ValidateEdits first; emission-order violations (a replacement
    /// input produced after its rewrite's emission point) are rejected there.
    /// Returns InvalidArgument if the result would be invalid.
    /// @param options Optional commit behavior; see CommitOptions.
    /// @return New ModelGraph owning the materialized result, or the first error.
    StatusOr<ModelGraph> Commit(const CommitOptions& options = {}) const;

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

    /// Metadata for a session-local constant added via AddSessionConstant.
    struct SessionConstant {
        TensorSpec spec{};
        ConstantBinding binding{};
        QuantizationSpec quantization{};
        std::string name{};
    };

    enum class ValueKind : std::uint8_t {
        kSource,
        kSessionConstant,
        kSessionVirtual,
        kInvalid,
    };

    /// Temporary mapping tables used during Commit() to translate value ids
    /// from source/session spaces into the committed graph's id space.
    using ValueMap = std::vector<std::optional<GraphValueId>>;
    struct CommitValueMaps {
        ValueMap& source_values;    // Source value -> committed value
        ValueMap& session_constants;// Session constant -> committed constant
        ValueMap& virtual_values;   // Virtual value -> committed output
    };

    struct RetainedNodes {
        std::vector<std::vector<uint8_t>> replacement_mask{};
        std::vector<uint8_t> source_node_mask{};
    };

    /// Cached consumer index for one mutation generation.
    /// Keys are resolved value ids, so ReplaceValue aliases share consumers.
    struct ConsumerCache {
        std::uint64_t generation = 0;
        std::vector<std::vector<GraphNodeId>> original_consumers{};
        std::vector<uint32_t> replacement_consumer_counts{};
    };

    // Validates that node is a source graph node id (ids below graph_.GetNodes().size()).
    // Returns InvalidArgument for out-of-range node ids.
    Status CheckSourceNodeId(GraphNodeId node) const;
    // Validates that value is a source graph value id (ValueKind::kSource only).
    // Returns InvalidArgument for session-local or invalid value ids.
    Status CheckSourceValueId(GraphValueId value) const;
    // Validates that value is a source or session constant value id
    // (ValueKind::kSource | ValueKind::kSessionConstant).
    // Returns InvalidArgument for session virtual or invalid value ids.
    Status CheckNonVirtualValueId(GraphValueId value) const;
    // Validates that value is an allocated, known value id (every kind except
    // ValueKind::kInvalid). Does not validate virtual production/order; that
    // remains the responsibility of ValidateVirtualValues().
    Status CheckKnownValueId(GraphValueId value) const;
    // Classifies a value id into the session's value-kind space:
    // kSource for ids below the source graph value range, kSessionConstant or
    // kSessionVirtual for ids in the session-local range (distinguished by the
    // presence of SessionConstant metadata), and kInvalid for ids beyond the
    // session-local range (never allocated by the session).
    AM_NODISCARD ValueKind ClassifyValue(GraphValueId value) const noexcept;
    // True when value is in the allocated session-local id space.
    AM_NODISCARD bool IsSessionLocalValue(GraphValueId value) const noexcept;
    // True when value is a session constant (session-local + SessionConstant metadata).
    AM_NODISCARD bool IsSessionConstant(GraphValueId value) const noexcept;
    // True when value is a session virtual value (session-local, not a session constant).
    AM_NODISCARD bool IsSessionVirtualValue(GraphValueId value) const noexcept;
    // True when any active rewrite's replacement output takes over this value.
    AM_NODISCARD bool IsValueReplacedByActiveRewrite(GraphValueId value) const noexcept;
    // Returns the 0-based index into the session value space.
    AM_NODISCARD std::size_t GetSessionValueIndex(GraphValueId session_value) const noexcept {
        return session_value.index - graph_.GetValues().size();
    }
    // Validates that replacement node inputs/outputs reference valid ids.
    Status ValidateReplacementNode(const ReplacementNode& replacement) const;
    // Validates that replacement targets belong to old_nodes and are not duplicated.
    Status ValidateReplacementTargets(
            std::span<const GraphNodeId> old_nodes,
            const std::vector<ReplacementNode>& replacement_nodes) const;
    // Validates virtual value ordering (no consumption before production).
    Status ValidateVirtualValues() const;
    // Resolves the TensorSpec of a value id known to the session.
    // Source values are read from graph_; session constants from session_value_metadata_;
    // virtual values must have an inferred spec in virtual_specs (caller-provided
    // scratch map). Returns NotFound for virtual values with no inferred spec.
    StatusOr<TensorSpec> ResolveValueSpec(
            GraphValueId value,
            const std::vector<std::optional<TensorSpec>>& virtual_specs) const;
    // Replays InferOperator over replacement nodes in order, verifying input
    // spec availability (dtype/shape resolvability), output count, replaces
    // target dtype compatibility, and deriving virtual specs into
    // virtual_specs_out. Emission-order availability of source value inputs is
    // NOT checked here; ValidateReplacementInputAvailability handles that.
    // Used by ValidateEdits and ReplaceSubgraph to catch semantic violations
    // before Commit.
    Status ValidateReplacementSemantics(
            const std::vector<ReplacementNode>& replacements,
            std::vector<std::optional<TensorSpec>>& virtual_specs_out) const;
    // Validates that every active rewrite's replacement inputs are available at
    // the rewrite's commit emission point. Commit emits each rewrite when the
    // first old_node appears in source topological order; a replacement input
    // referencing a value whose producer is emitted later would fail to map
    // during commit. Session constants, virtual values, and producer-less
    // source values (inputs/weights/constants/states) are always available.
    // A producer covered by a rewrite must have that rewrite bind the value as
    // a replacement output (otherwise the value is never emitted, e.g. a
    // RemoveNode rewrite); a producer inside the consuming rewrite itself must
    // be bound by an earlier replacement in the same group. Returns
    // InvalidArgument when a source-value input is produced by a node or
    // rewrite that emits at or after the consuming rewrite's emission point.
    Status ValidateReplacementInputAvailability() const;
    // Builds a GraphValueDesc from a session constant's metadata.
    AM_NODISCARD GraphValueDesc MakeOutputDescFromSessionConstant(GraphValueId value) const;
    // Translates a value id from source/session space to committed graph space.
    StatusOr<GraphValueId> MapCommittedValue(
            GraphValueId value,
            const CommitValueMaps& maps) const;
    // Computes the mixed source/replacement reverse-reachability closure used
    // by Commit pruning.
    StatusOr<RetainedNodes> ComputeRetainedNodes() const;
    // Copies source external values and session constants into the committed
    // graph; session constants referenced only by unretained nodes/outputs
    // are dropped in lockstep with node pruning (aligned with AddSessionConstant).
    Status CopyExternalValues(ModelGraph& committed,
                              const CommitValueMaps& maps,
                              const RetainedNodes& retained) const;
    // Emits all replacement nodes in a rewrite entry into the committed graph.
    Status EmitRewrite(const RewriteEntry& rewrite,
                       std::span<const uint8_t> retained_mask,
                       ModelGraph& committed,
                       const CommitValueMaps& maps) const;
    // Emits a single surviving original node into the committed graph.
    Status EmitOriginalNode(GraphNodeId old_node,
                            ModelGraph& committed,
                            const CommitValueMaps& maps) const;
    // Maps graph output values through resolution into committed graph.
    Status MarkCommittedOutputs(ModelGraph& committed,
                                const CommitValueMaps& maps) const;
    // Marks a rewrite as inactive; clears node_to_rewrite_ entries.
    void DeactivateRewrite(std::size_t rewrite_index);
    // Invalidates cached consumer indexes after a successful state mutation.
    void InvalidateConsumerCache() noexcept;
    // Builds or returns the consumer index for the current mutation generation.
    // Returns InvalidArgument if the source graph contains a cycle (the session
    // does not introduce new edges, so a cycle can only originate from the
    // source graph). The returned pointer is valid only until the next
    // mutation invalidates the cache.
    StatusOr<const ConsumerCache*> EnsureConsumerCache() const;

    // Immutable source graph; must outlive the session.
    const ModelGraph& graph_;
    // Per-session-value metadata and authoritative count:
    // has_value = session constant, nullopt = virtual.
    // Index i corresponds to session value at graph_.GetValues().size() + i.
    std::vector<std::optional<SessionConstant>> session_value_metadata_{};
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
    bool commit_pruning_requested_ = false;
};

/// @brief Convenience builder for constructing subgraph replacements without
/// manually managing virtual value allocation, RewriteOutputBinding wiring,
/// and ReplacementNode assembly.
///
/// Emit allocates virtual values for internal edges and returns their ids.
/// Yield redirects a virtual value (returned by Emit) to replace a real
/// source graph value. Yield also redirects every pending consumer that
/// still references the virtual value, and later Emit calls normalize such
/// references automatically, so a yielded value stays usable both as an
/// input to new nodes and as a replacement target. Commit submits the
/// accumulated replacement nodes to the session as a single
/// ReplaceSubgraph mutation.
///
/// Usage:
///   SubgraphBuilder builder(session, {old_node1, old_node2});
///   AM_ASSIGN_OR_RETURN(GraphValueId mid,
///                        builder.Emit(OpType::kSilu, {input}, output_desc, params));
///   AM_ASSIGN_OR_RETURN(GraphValueId out,
///                        builder.Emit(OpType::kMul, {mid, other}, output_desc, params));
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

    /// @brief Creates a new replacement node with a single output described by
    /// `output_desc`. Allocates a virtual value internally, binds it as the
    /// node's output, and returns the virtual value id for use as input to
    /// subsequent Emit calls or as the internal_val argument to Yield.
    /// @param op_type Operator type of the new node.
    /// @param inputs Input value ids for the new node.
    /// @param output_desc Descriptor for the single output.
    /// @param op_params Operator-specific parameters.
    /// @param decoder_layer_index Optional decoder layer index tag.
    /// @param name Optional debug name for the new node.
    /// @return Virtual value id bound to the new node's output.
    StatusOr<GraphValueId> Emit(
            OpType op_type,
            std::vector<GraphValueId> inputs,
            NodeOutputDesc output_desc,
            OpParams op_params = std::monostate{},
            std::optional<uint32_t> decoder_layer_index = std::nullopt,
            std::string name = {});

    /// @brief Creates a new replacement node with one output per descriptor in
    /// `output_descs`. Allocates one virtual value per output and returns the
    /// virtual values in descriptor order.
    /// @param op_type Operator type of the new node.
    /// @param inputs Input value ids for the new node.
    /// @param output_descs Descriptors for each output.
    /// @param op_params Operator-specific parameters.
    /// @param decoder_layer_index Optional decoder layer index tag.
    /// @param name Optional debug name for the new node.
    /// @return Virtual value ids bound to the new node's outputs, in descriptor order.
    StatusOr<std::vector<GraphValueId>> Emit(
            OpType op_type,
            std::vector<GraphValueId> inputs,
            std::vector<NodeOutputDesc> output_descs,
            OpParams op_params = std::monostate{},
            std::optional<uint32_t> decoder_layer_index = std::nullopt,
            std::string name = {});

    /// @brief Marks an internal virtual value (returned by Emit) as the replacement
    /// for an external graph value. After Commit, all consumers of
    /// old_value_to_replace will consume the new node's output instead.
    ///
    /// Redirects every pending replacement input that still references
    /// internal_val to old_value_to_replace, and records the alias so that
    /// later Emit calls normalize inputs referencing internal_val as well.
    ///
    /// Returns InvalidArgument if internal_val was not produced by any
    /// prior Emit call, if it was already yielded, or if
    /// old_value_to_replace is not produced by any node in old_nodes.
    /// @param internal_val Virtual value id returned by a prior Emit call.
    /// @param old_value_to_replace External graph value to take over.
    /// @return Status::Ok() on success, or InvalidArgument if internal_val is
    ///         unknown or already yielded, or if old_value_to_replace is not
    ///         produced by any node in old_nodes.
    Status Yield(GraphValueId internal_val, GraphValueId old_value_to_replace);

    /// @brief Discards all pending Emit/Yield state, restoring the builder to
    /// its construction-time state while keeping the same old_nodes.
    ///
    /// Commit leaves new_nodes_/aliases_ in place on failure so the failure
    /// can be diagnosed. Reset() drops that state, allowing a corrected
    /// Emit/Yield/Commit cycle to be retried on the same replacement target
    /// set without recreating the builder. The session is unaffected: any
    /// prior successful Commit's effects remain committed.
    void Reset() noexcept;

    /// @brief Submits the accumulated replacement nodes to the session as a single
    /// ReplaceSubgraph mutation. On success, the builder is reset and can be
    /// reused for further Emit/Yield/Commit cycles. On failure, pending state is
    /// retained for diagnosis; call Reset() to clear it before retrying.
    /// @return Status::Ok() on success, or the first error from the underlying mutation.
    Status Commit();

private:
    GraphRewriteSession& session_;
    std::vector<GraphNodeId> old_nodes_;
    std::vector<ReplacementNode> new_nodes_;

    /// @brief Hash functor for GraphValueId keys (no std::hash specialization
    ///        exists for GraphValueId).
    struct ValueIdHash {
        std::size_t operator()(GraphValueId id) const noexcept {
            return std::hash<uint32_t>{}(id.index);
        }
    };

    /// @brief Yields recorded so far: virtual value id -> replacement target
    ///        (always a source value id, enforced by Yield). Emit normalizes
    ///        new inputs through this map, Yield rewrites existing inputs via
    ///        NormalizeInputs, and Commit clears the map on success so the
    ///        builder can be reused.
    std::unordered_map<GraphValueId, GraphValueId, ValueIdHash> aliases_;

    /// @brief Redirects every pending replacement input referencing `from` to
    ///        `to`. Used by Yield for immediate redirection and by Commit as
    ///        a final normalization pass.
    void NormalizeInputs(GraphValueId from, GraphValueId to) noexcept;
};

}// namespace aethermind

#endif
