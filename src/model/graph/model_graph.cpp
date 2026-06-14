#include "aethermind/model/graph/model_graph.h"

#include <utility>

namespace aethermind {

ModelGraph::ModelGraph(HfModelConfig config, std::vector<GraphNode> nodes) noexcept
    : config_(std::move(config)), nodes_(std::move(nodes)) {}

std::span<const GraphNode> ModelGraph::GetNodes() const noexcept {
    return nodes_;
}

const HfModelConfig& ModelGraph::GetConfig() const noexcept {
    return config_;
}

}// namespace aethermind
