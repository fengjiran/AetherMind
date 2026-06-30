#include "aethermind/model/graph/graph_op_builder.h"

#include <utility>
#include <vector>

namespace aethermind {
namespace {

GraphValueId OnlyOneOutput(const AddedNode& added_node) {
    AM_CHECK(added_node.outputs.size() == 1U,
             "Expected graph op builder helper to create exactly one output");
    return added_node.outputs.front();
}

NodeOutputDesc ActivationOutput(TensorSpec spec) {
    return NodeOutputDesc{.spec = std::move(spec), .payload = ActivationValue{}};
}

}// namespace

GraphValueId AddInput(ModelGraph& graph, TensorSpec spec, std::string debug_name) {
    return graph.AddInput(std::move(spec), std::move(debug_name));
}

GraphValueId AddState(ModelGraph& graph,
                      TensorSpec spec,
                      StateBinding binding,
                      std::string debug_name) {
    return graph.AddState(std::move(spec), binding, std::move(debug_name));
}

GraphValueId AddLinear(ModelGraph& graph,
                       GraphValueId input,
                       int64_t out_features,
                       DataType weight_dtype,
                       WeightBinding binding,
                       std::string debug_name) {
    AM_CHECK(out_features > 0, "Linear out_features must be positive");

    // Snapshot input specs BEFORE graph mutation — graph.AddWeight/AddNode may
    // reallocate the graph's value storage and invalidate references into it.
    const TensorSpec input_spec = graph.GetValue(input).spec;
    AM_CHECK(input_spec.shape.IsRanked(), "Linear input shape must be ranked");
    std::vector<ShapeSymbol> input_shape = *input_spec.shape.shape();
    AM_CHECK(!input_shape.empty(), "Linear input rank must be at least 1");

    const ShapeSymbol in_features_symbol = input_shape.back();
    AM_CHECK(in_features_symbol.IsStatic(), "Linear input last dimension must be static");
    AM_CHECK(in_features_symbol.GetStaticValue() > 0, "Linear input last dimension must be positive");

    const ShapeSymbol out_features_symbol = ShapeSymbol::CreateFromValue(out_features);
    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {out_features_symbol, in_features_symbol}},
            binding,
            debug_name);

    input_shape.back() = out_features_symbol;
    const auto node = graph.AddNode(
            OpType::kLinear,
            binding.decoder_layer_index,
            {input, weight},
            {ActivationOutput(
                    {.dtype = input_spec.dtype,
                     .shape = SymbolicShape(std::move(input_shape))})},
            LinearParams{},
            {},
            std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddRmsNorm(ModelGraph& graph,
                        GraphValueId input,
                        DataType weight_dtype,
                        WeightBinding binding,
                        float eps,
                        std::string debug_name) {
    const TensorSpec input_spec = graph.GetValue(input).spec;
    AM_CHECK(input_spec.shape.IsRanked(), "RmsNorm input shape must be ranked");
    const std::vector<ShapeSymbol> input_shape = *input_spec.shape.shape();
    AM_CHECK(!input_shape.empty(), "RmsNorm input rank must be at least 1");

    const ShapeSymbol in_features_symbol = input_shape.back();
    AM_CHECK(in_features_symbol.IsStatic(), "RmsNorm input last dimension must be static");
    AM_CHECK(in_features_symbol.GetStaticValue() > 0, "RmsNorm input last dimension must be positive");

    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {in_features_symbol}},
            binding,
            debug_name);

    const auto node = graph.AddNode(
            OpType::kRmsNorm,
            binding.decoder_layer_index,
            {input, weight},
            {ActivationOutput(input_spec)},
            RmsNormParams{.eps = eps},
            {},
            std::move(debug_name));
    return OnlyOneOutput(node);
}

GraphValueId AddEmbedding(ModelGraph& graph,
                          GraphValueId token_ids,
                          int64_t vocab_size,
                          int64_t embedding_dim,
                          DataType weight_dtype,
                          WeightBinding binding,
                          std::string debug_name) {
    AM_CHECK(vocab_size > 0, "Embedding vocab_size must be positive");
    AM_CHECK(embedding_dim > 0, "Embedding embedding_dim must be positive");

    const TensorSpec token_spec = graph.GetValue(token_ids).spec;
    AM_CHECK(token_spec.shape.IsRanked(), "Embedding token_ids shape must be ranked");
    std::vector<ShapeSymbol> output_shape = *token_spec.shape.shape();
    output_shape.push_back(ShapeSymbol::CreateFromValue(embedding_dim));

    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {ShapeSymbol::CreateFromValue(vocab_size),
                       ShapeSymbol::CreateFromValue(embedding_dim)}},
            binding,
            debug_name);

    const auto node = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {token_ids, weight},
            {ActivationOutput(
                    {.dtype = weight_dtype,
                     .shape = SymbolicShape(std::move(output_shape))})},
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

KVCachePair AddKVCacheUpdate(ModelGraph& graph,
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
            {NodeOutputDesc{.spec = std::move(k_output_spec),
                            .payload = StateValue{.binding = k_binding}},
             NodeOutputDesc{.spec = std::move(v_output_spec),
                            .payload = StateValue{.binding = v_binding}}},
            KVCacheUpdateParams{},
            {},
            std::move(debug_name));
    AM_CHECK(node.outputs.size() == 2U,
             "Expected KV cache update helper to create exactly two outputs");
    return KVCachePair{.k = node.outputs[0], .v = node.outputs[1]};
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
