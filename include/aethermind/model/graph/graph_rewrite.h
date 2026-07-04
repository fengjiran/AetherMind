#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H

#include "aethermind/model/graph/graph.h"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace aethermind {

struct NodeRemoval {
    GraphNodeId node{};
};

struct InputRedirection {
    GraphNodeId node{};
    size_t input_index = 0;
    GraphValueId new_value{};
};

struct ValueReplacement {
    GraphValueId old_value{};
    GraphValueId new_value{};
};

struct RewriteOutputBinding {
    NodeOutputDesc desc{};
    std::optional<GraphValueId> replaces{};
};

struct ReplacementNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<RewriteOutputBinding> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};
    std::string debug_name{};
};

struct SubgraphReplacement {
    std::vector<GraphNodeId> old_nodes{};
    std::vector<ReplacementNode> replacement_nodes{};
};

using GraphMutation = std::variant<SubgraphReplacement,
                                   NodeRemoval,
                                   InputRedirection,
                                   ValueReplacement>;

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

class GraphRewriteSession {
public:
    explicit GraphRewriteSession(const ModelGraph& graph);
    AM_NODISCARD GraphValueId AllocateVirtualValue();

    /// Applies a batch of mutations sequentially. Non-atomic: if mutation N
    /// fails, mutations 0..N-1 remain applied to the session. The caller is
    /// responsible for either committing the partial state or discarding the
    /// session.
    AM_NODISCARD Status Apply(std::span<const GraphMutation> mutations);
    AM_NODISCARD Status RemoveNode(GraphNodeId node);
    AM_NODISCARD Status ReplaceSubgraph(std::span<const GraphNodeId> old_nodes,
                                        const std::vector<ReplacementNode>& replacement_nodes);
    AM_NODISCARD Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    AM_NODISCARD Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);
    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;
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

    AM_NODISCARD Status ValidateEdits() const;
    AM_NODISCARD StatusOr<ModelGraph> Commit() const;

    /// Validates that `value` refers to a real (non-virtual) graph value.
    /// Returns InvalidArgument if the id is out of range or refers to a
    /// virtual value allocated by AllocateVirtualValue().
    AM_NODISCARD Status CheckValueId(GraphValueId value) const;

private:
    struct RewriteEntry {
        std::vector<GraphNodeId> old_nodes{};
        std::vector<ReplacementNode> replacements{};
        bool active = true;
        bool exposes_node_view = false;
    };

    AM_NODISCARD Status CheckNodeId(GraphNodeId node) const;
    AM_NODISCARD Status CheckValueIdAllowVirtual(GraphValueId value) const;
    AM_NODISCARD bool IsVirtualValue(GraphValueId value) const noexcept;
    AM_NODISCARD std::size_t GetVirtualIndex(GraphValueId virtual_id) const noexcept {
        return virtual_id.index - graph_.GetValues().size();
    }
    AM_NODISCARD Status ValidateReplacementNode(const ReplacementNode& replacement) const;
    AM_NODISCARD Status ValidateVirtualValues() const;
    void DeactivateRewrite(std::size_t rewrite_index);

    const ModelGraph& graph_;
    std::size_t virtual_value_count_ = 0;
    std::vector<RewriteEntry> rewrites_{};
    std::vector<std::optional<std::size_t>> node_to_rewrite_{};
    std::vector<std::optional<GraphValueId>> value_replacements_{};
    mutable std::vector<std::optional<GraphValueId>> resolved_value_cache_{};
};

/// Convenience builder for constructing subgraph replacements without
/// manually managing virtual value allocation, RewriteOutputBinding wiring,
/// and ReplacementNode assembly.
///
/// Usage:
///   SubgraphBuilder builder(session, {old_node1, old_node2});
///   GraphValueId mid = builder.Emit(OpType::kSilu, {input}, spec, params);
///   GraphValueId out = builder.Emit(OpType::kMul, {mid, other}, spec, params);
///   builder.Yield(out, old_output_value);
///   builder.Commit();
///
/// Thread-safety: not thread-safe. Each builder instance is intended for
/// single-threaded use within one pass invocation.
class SubgraphBuilder {
public:
    SubgraphBuilder(GraphRewriteSession& session, std::vector<GraphNodeId> old_nodes)
        : session_(session), old_nodes_(std::move(old_nodes)) {}

    /// Creates a new replacement node with a single activation output.
    /// Allocates a virtual value internally, binds it as the node's output,
    /// and returns the virtual value id for use as input to subsequent Emit
    /// calls or as the internal_val argument to Yield.
    ///
    /// For nodes that need multiple outputs or non-activation payloads
    /// (e.g. StateValue for KV cache updates), use the full
    /// GraphRewriteSession::ReplaceSubgraph API directly.
    GraphValueId Emit(OpType op_type,
                      std::vector<GraphValueId> inputs,
                      TensorSpec output_spec,
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
