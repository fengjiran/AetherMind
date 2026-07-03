#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_REWRITE_H

#include "aethermind/model/graph/graph.h"

#include <cstddef>
#include <optional>
#include <span>
#include <variant>

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
    AM_NODISCARD Status Apply(std::span<const GraphMutation> mutations);
    AM_NODISCARD Status RemoveNode(GraphNodeId node);
    AM_NODISCARD Status ReplaceSubgraph(std::span<const GraphNodeId> old_nodes,
                                        const std::vector<ReplacementNode>& replacement_nodes);
    AM_NODISCARD Status RedirectInput(GraphNodeId node, size_t input_index, GraphValueId new_value);
    AM_NODISCARD Status ReplaceValue(GraphValueId old_value, GraphValueId new_value);
    AM_NODISCARD GraphValueId GetResolvedValue(GraphValueId value) const;
    AM_NODISCARD StatusOr<GraphNodeView> GetNodeView(GraphNodeId node) const;
    AM_NODISCARD Status ValidateEdits() const;
    AM_NODISCARD StatusOr<ModelGraph> Commit() const;

private:
    struct RewriteEntry {
        std::vector<GraphNodeId> old_nodes{};
        std::vector<ReplacementNode> replacements{};
        bool active = true;
        bool exposes_node_view = false;
    };

    AM_NODISCARD Status CheckNodeId(GraphNodeId node) const;
    AM_NODISCARD Status CheckValueId(GraphValueId value) const;
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

}// namespace aethermind

#endif
