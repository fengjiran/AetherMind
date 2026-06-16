#ifndef AETHERMIND_MODEL_GRAPH_MODEL_GRAPH_H
#define AETHERMIND_MODEL_GRAPH_MODEL_GRAPH_H

#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/operators/operator.h"
#include "aethermind/runtime/workspace.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace aethermind {

enum class ModelWeightRole : uint8_t {
    kTokenEmbedding,
    kAttentionQ,
    kAttentionK,
    kAttentionV,
    kAttentionO,
    kMlpGate,
    kMlpUp,
    kMlpDown,
    kInputNorm,
    kPostAttentionNorm,
    kFinalNorm,
    kLmHead,
};

struct ModelWeightBinding {
    ModelWeightRole role{};
    std::optional<uint32_t> decoder_layer_index{};
};

struct ModelGraphAttrs {
    std::vector<std::byte> bytes{};
};

struct GraphNode {
    OpType op_type = OpType::kUnknown;
    std::optional<uint32_t> decoder_layer_index{};
    std::vector<TensorSpec> inputs{};
    std::vector<TensorSpec> outputs{};
    std::vector<ModelWeightBinding> weights{};
    ModelGraphAttrs attrs{};
    std::any op_params{};
    WorkspaceRequirement workspace_requirement{};
};

class ModelGraph {
public:
    ModelGraph() = default;
    ModelGraph(HfModelConfig config, std::vector<GraphNode> nodes) noexcept;

    AM_NODISCARD std::span<const GraphNode> GetNodes() const noexcept;
    AM_NODISCARD const HfModelConfig& GetConfig() const noexcept;

private:
    HfModelConfig config_{};
    std::vector<GraphNode> nodes_{};
};

}// namespace aethermind

#endif
