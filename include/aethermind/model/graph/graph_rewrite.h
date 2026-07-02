#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H

#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/graph_types.h"

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

using GraphMutation = std::variant<ReplaceNodeCmd,
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
    AM_NODISCARD Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    AM_NODISCARD Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);

    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;
    AM_NODISCARD StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;

    AM_NODISCARD Status ValidateEdits() const;
    AM_NODISCARD StatusOr<ModelGraph> Commit() const;

private:
    AM_NODISCARD Status CheckNodeId(GraphNodeId node) const;
    AM_NODISCARD Status CheckValueId(GraphValueId value) const;
    AM_NODISCARD const std::vector<GraphValueId>& CurrentInputs(GraphNodeId node) const;

    const ModelGraph& graph_;
    std::vector<bool> removed_nodes_{};
    std::vector<std::optional<GraphValueId>> value_replacements_{};
    mutable std::vector<std::optional<GraphValueId>> resolved_value_cache_{};
    std::vector<std::optional<std::vector<GraphValueId>>> input_overrides_{};
    std::vector<std::optional<std::vector<GraphNode>>> node_replacements_{};
};

}// namespace aethermind

#endif
