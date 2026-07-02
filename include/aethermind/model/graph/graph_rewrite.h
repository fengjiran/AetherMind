#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H

#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/graph_types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace aethermind {

struct ReplaceNodeCmd {
    GraphNodeId old_node{};
    std::vector<GraphNode> replacement_nodes{};
};

struct RemoveNodeCmd {
    GraphNodeId node{};
};

struct RedirectInputCmd {
    GraphNodeId node{};
    size_t input_index = 0;
    GraphValueId new_value{};
};

struct ReplaceValueCmd {
    GraphValueId old_value{};
    GraphValueId new_value{};
};

struct ReplacementOutput {
    NodeOutputDesc desc{};
    std::optional<GraphValueId> replaces{};
};

struct ReplacementNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<GraphValueId> inputs{};
    std::vector<ReplacementOutput> outputs{};
    ModelGraphAttrs attrs{};
    OpParams op_params{};
    std::string debug_name{};
};

struct ReplaceSubgraphCmd {
    std::vector<GraphNodeId> old_nodes{};
    std::vector<ReplacementNode> replacement_nodes{};
};

using GraphMutation = std::variant<ReplaceNodeCmd,
                                   ReplaceSubgraphCmd,
                                   RemoveNodeCmd,
                                   RedirectInputCmd,
                                   ReplaceValueCmd>;

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

    AM_NODISCARD Status Apply(std::span<const GraphMutation> mutations);

    AM_NODISCARD Status RemoveNode(GraphNodeId node);
    AM_NODISCARD Status ReplaceNode(GraphNodeId node, const std::vector<GraphNode>& replacement_nodes);
    AM_NODISCARD Status ReplaceNodeWithOutputs(GraphNodeId node, const std::vector<ReplacementNode>& replacement_nodes);
    AM_NODISCARD Status ReplaceSubgraph(std::span<const GraphNodeId> old_nodes, const std::vector<ReplacementNode>& replacement_nodes);
    AM_NODISCARD Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    AM_NODISCARD Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);

    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;
    AM_NODISCARD StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;

    AM_NODISCARD Status ValidateEdits() const;
    AM_NODISCARD StatusOr<ModelGraph> Commit() const;

private:
    enum class NodeRewriteKind : std::uint8_t {
        kKeep,
        kRemove,
    };

    struct NodeRewriteEntry {
        NodeRewriteKind kind = NodeRewriteKind::kKeep;
        std::optional<std::size_t> subgraph_rewrite{};
    };

    struct SubgraphRewriteEntry {
        std::vector<GraphNodeId> old_nodes{};
        std::vector<ReplacementNode> replacements{};
    };

    AM_NODISCARD Status CheckNodeId(GraphNodeId node) const;
    AM_NODISCARD Status CheckValueId(GraphValueId value) const;
    AM_NODISCARD Status ValidateReplacementNode(const ReplacementNode& replacement) const;
    void ClearSubgraphRewrite(std::size_t subgraph_index);
    AM_NODISCARD const std::vector<GraphValueId>& CurrentInputs(GraphNodeId node) const;

    const ModelGraph& graph_;
    std::vector<NodeRewriteEntry> node_rewrites_{};
    std::vector<SubgraphRewriteEntry> subgraph_rewrites_{};
    std::vector<std::optional<GraphValueId>> value_replacements_{};
    mutable std::vector<std::optional<GraphValueId>> resolved_value_cache_{};
    std::vector<std::optional<std::vector<GraphValueId>>> input_overrides_{};
};

}// namespace aethermind

#endif
