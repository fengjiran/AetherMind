#include "aethermind/model/graph/graph_op_builder.h"

#include <utility>

namespace aethermind {
namespace {

GraphValueId OnlyOneOutput(const ModelGraph::AddedNode& added_node) {
    AM_CHECK(added_node.outputs.size() == 1U,
             "Expected graph op builder helper to create exactly one output");
    return added_node.outputs.front();
}

ModelGraph::NodeOutputDesc ActivationOutput(TensorSpec spec) {
    return ModelGraph::NodeOutputDesc{.spec = std::move(spec), .payload = ActivationValue{}};
}

}// namespace

GraphValueId AddLinear(ModelGraph& graph,
                       std::optional<uint32_t> decoder_layer_index,
                       GraphValueId input,
                       GraphValueId weight,
                       TensorSpec output_spec,
                       std::string debug_name) {
    const auto node = graph.AddNode(
            OpType::kLinear,
            decoder_layer_index,
            {input, weight},
            {ActivationOutput(std::move(output_spec))},
            LinearParams{},
            {},
            std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddRmsNorm(ModelGraph& graph,
                        std::optional<uint32_t> decoder_layer_index,
                        GraphValueId input,
                        GraphValueId weight,
                        TensorSpec output_spec,
                        float eps,
                        std::string debug_name) {
    const auto node = graph.AddNode(
            OpType::kRmsNorm,
            decoder_layer_index,
            {input, weight},
            {ActivationOutput(std::move(output_spec))},
            RmsNormParams{.eps = eps},
            {},
            std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddEmbedding(ModelGraph& graph,
                          GraphValueId token_ids,
                          GraphValueId weight,
                          TensorSpec output_spec,
                          std::string debug_name) {
    const auto node = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {token_ids, weight},
            {ActivationOutput(std::move(output_spec))},
            EmbeddingParams{},
            {},
            std::move(debug_name));
    return OnlyOneOutput(node);
}

RoPEOutputs AddRoPE(ModelGraph& graph,
                    std::optional<uint32_t> decoder_layer_index,
                    GraphValueId q,
                    GraphValueId k,
                    GraphValueId position_ids,
                    TensorSpec q_output_spec,
                    TensorSpec k_output_spec,
                    RoPEParams params,
                    std::string debug_name) {
    const auto node = graph.AddNode(
            OpType::kRoPE,
            decoder_layer_index,
            {q, k, position_ids},
            {ActivationOutput(std::move(q_output_spec)),
             ActivationOutput(std::move(k_output_spec))},
            params,
            {},
            std::move(debug_name));
    AM_CHECK(node.outputs.size() == 2U, "Expected RoPE helper to create exactly two outputs");
    return RoPEOutputs{.q = node.outputs[0], .k = node.outputs[1]};
}

KVCacheUpdateOutputs AddKVCacheUpdate(ModelGraph& graph,
                                      std::optional<uint32_t> decoder_layer_index,
                                      GraphValueId k_new,
                                      GraphValueId v_new,
                                      GraphValueId k_cache,
                                      GraphValueId v_cache,
                                      TensorSpec k_output_spec,
                                      TensorSpec v_output_spec,
                                      StateBinding k_binding,
                                      StateBinding v_binding,
                                      std::string debug_name) {
    const auto node = graph.AddNode(
            OpType::kKVCacheUpdate,
            decoder_layer_index,
            {k_new, v_new, k_cache, v_cache},
            {ModelGraph::NodeOutputDesc{.spec = std::move(k_output_spec),
                                        .payload = StateValue{.binding = k_binding}},
             ModelGraph::NodeOutputDesc{.spec = std::move(v_output_spec),
                                        .payload = StateValue{.binding = v_binding}}},
            KVCacheUpdateParams{},
            {},
            std::move(debug_name));
    AM_CHECK(node.outputs.size() == 2U,
             "Expected KV cache update helper to create exactly two outputs");
    return KVCacheUpdateOutputs{.k = node.outputs[0], .v = node.outputs[1]};
}

GraphValueId AddAttention(ModelGraph& graph,
                          std::optional<uint32_t> decoder_layer_index,
                          GraphValueId q,
                          GraphValueId k,
                          GraphValueId v,
                          TensorSpec output_spec,
                          AttentionParams params,
                          std::string debug_name) {
    const auto node = graph.AddNode(OpType::kAttention,
                                    decoder_layer_index,
                                    {q, k, v},
                                    {ActivationOutput(std::move(output_spec))},
                                    params,
                                    {},
                                    std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddElementwiseAdd(ModelGraph& graph,
                               std::optional<uint32_t> decoder_layer_index,
                               GraphValueId lhs,
                               GraphValueId rhs,
                               TensorSpec output_spec,
                               std::string debug_name) {
    const auto node = graph.AddNode(OpType::kAdd,
                                    decoder_layer_index,
                                    {lhs, rhs},
                                    {ActivationOutput(std::move(output_spec))},
                                    AddParams{},
                                    {},
                                    std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddSiluMul(ModelGraph& graph,
                        std::optional<uint32_t> decoder_layer_index,
                        GraphValueId gate,
                        GraphValueId up,
                        TensorSpec output_spec,
                        std::string debug_name) {
    const auto node = graph.AddNode(OpType::kSiluMul,
                                    decoder_layer_index,
                                    {gate, up},
                                    {ActivationOutput(std::move(output_spec))},
                                    SiluMulParams{},
                                    {},
                                    std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddArgmax(ModelGraph& graph,
                       std::optional<uint32_t> decoder_layer_index,
                       GraphValueId input,
                       TensorSpec output_spec,
                       int64_t axis,
                       std::string debug_name) {
    const auto node = graph.AddNode(OpType::kArgmax,
                                    decoder_layer_index,
                                    {input},
                                    {ActivationOutput(std::move(output_spec))},
                                    ArgmaxParams{.axis = axis},
                                    {},
                                    std::move(debug_name));
    return OnlyOneOutput(node);
}

}// namespace aethermind
