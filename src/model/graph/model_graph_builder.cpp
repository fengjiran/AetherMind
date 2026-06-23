#include "aethermind/model/graph/model_graph_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/operators/embedding_op.h"
#include "aethermind/operators/rmsnorm_op.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace aethermind {
namespace {

// Decoder-only dense Llama layer nodes, in execution order:
// input RMSNorm, Q/K/V projection, RoPE, attention score MatMul, Softmax,
// attention value MatMul, output projection, residual Add, post-attention RMSNorm,
// MLP gate/up projection, SiLU multiply, MLP down projection, residual Add.
constexpr size_t kDenseLlamaLayerNodeCount = 16;

TensorSpec SymbolicTensorSpec(DataType dtype, std::vector<ShapeSymbol> dims) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(std::move(dims))};
}

std::vector<ShapeSymbol> SymbolicDims(const std::vector<int64_t>& shape) {
    std::vector<ShapeSymbol> dims;
    dims.reserve(shape.size());
    for (const int64_t dim: shape) {
        dims.push_back(ShapeSymbol::CreateFromValue(dim));
    }
    return dims;
}

TensorSpec WeightTensor(const RawWeightView& weight) {
    return SymbolicTensorSpec(weight.dtype, SymbolicDims(weight.shape));
}

TensorSpec ActivationTensor(DataType dtype, ShapeSymbol seq_len, int64_t feature_dim) {
    return SymbolicTensorSpec(dtype, {seq_len, ShapeSymbol::CreateFromValue(feature_dim)});
}

ModelWeightBinding Bind(ModelWeightRole role, std::optional<uint32_t> decoder_layer_index = std::nullopt) noexcept {
    return ModelWeightBinding{.role = role, .decoder_layer_index = decoder_layer_index};
}

struct LayerTensorSpecs {
    const TensorSpec& hidden;
    const TensorSpec& kv_hidden;
    const TensorSpec& intermediate;
    const TensorSpec& scores;
};

struct WeightedNodeSpecs {
    TensorSpec weight;
    TensorSpec output;
};

GraphValueId OnlyOutput(const ModelGraph::AddedNode& added_node) {
    AM_CHECK(added_node.outputs.size() == 1U, "Expected node to have exactly one output");
    return added_node.outputs.front();
}

ModelGraph::AddedNode AddWeightedNode(ModelGraph& graph,
                                      OpType op_type,
                                      std::optional<uint32_t> decoder_layer_index,
                                      std::vector<GraphValueId> inputs,
                                      WeightedNodeSpecs specs,
                                      ModelWeightBinding weight) {
    inputs.push_back(graph.AddWeight(std::move(specs.weight), weight));
    return graph.AddNode(op_type,
                         decoder_layer_index,
                         std::move(inputs),
                         {ModelGraph::NodeOutputDecl{.spec = std::move(specs.output), .payload = ActivationValue{}}});
}

ModelGraph::AddedNode AddPureNode(ModelGraph& graph,
                                  OpType op_type,
                                  std::optional<uint32_t> decoder_layer_index,
                                  std::vector<GraphValueId> inputs,
                                  std::vector<TensorSpec> outputs) {
    std::vector<ModelGraph::NodeOutputDecl> output_decls;
    output_decls.reserve(outputs.size());
    for (TensorSpec& output: outputs) {
        output_decls.push_back(ModelGraph::NodeOutputDecl{.spec = std::move(output), .payload = ActivationValue{}});
    }
    return graph.AddNode(op_type, decoder_layer_index, std::move(inputs), std::move(output_decls));
}

ModelGraph::AddedNode AddRmsNormNode(ModelGraph& graph,
                                     std::optional<uint32_t> decoder_layer_index,
                                     GraphValueId input,
                                     WeightedNodeSpecs specs,
                                     float eps,
                                     ModelWeightBinding weight) {
    GraphValueId weight_value = graph.AddWeight(std::move(specs.weight), weight);
    return graph.AddNode(OpType::kRmsNorm,
                         decoder_layer_index,
                         {input, weight_value},
                         {ModelGraph::NodeOutputDecl{.spec = std::move(specs.output), .payload = ActivationValue{}}},
                         RmsNormOp::Params{.eps = eps});
}

GraphValueId AppendDenseLlamaLayerNodes(ModelGraph& graph,
                                        uint32_t layer_index,
                                        const DecoderLayerRawWeights& layer,
                                        GraphValueId hidden,
                                        LayerTensorSpecs specs,
                                        float rms_norm_eps) {
    const size_t layer_begin = graph.GetNodes().size();

    const GraphValueId normed = OnlyOutput(AddRmsNormNode(graph,
                                                          layer_index,
                                                          hidden,
                                                          WeightedNodeSpecs{.weight = WeightTensor(layer.norm.input_rmsnorm), .output = specs.hidden},
                                                          rms_norm_eps,
                                                          Bind(ModelWeightRole::kInputNorm, layer_index)));
    const GraphValueId q = OnlyOutput(AddWeightedNode(graph,
                                                      OpType::kLinear,
                                                      layer_index,
                                                      {normed},
                                                      WeightedNodeSpecs{.weight = WeightTensor(layer.attn.q_proj), .output = specs.hidden},
                                                      Bind(ModelWeightRole::kAttentionQ, layer_index)));
    const GraphValueId k = OnlyOutput(AddWeightedNode(graph,
                                                      OpType::kLinear,
                                                      layer_index,
                                                      {normed},
                                                      WeightedNodeSpecs{.weight = WeightTensor(layer.attn.k_proj), .output = specs.kv_hidden},
                                                      Bind(ModelWeightRole::kAttentionK, layer_index)));
    const GraphValueId v = OnlyOutput(AddWeightedNode(graph,
                                                      OpType::kLinear,
                                                      layer_index,
                                                      {normed},
                                                      WeightedNodeSpecs{.weight = WeightTensor(layer.attn.v_proj), .output = specs.kv_hidden},
                                                      Bind(ModelWeightRole::kAttentionV, layer_index)));
    const ModelGraph::AddedNode rope = AddPureNode(graph, OpType::kRoPE, layer_index, {q, k}, {specs.hidden, specs.kv_hidden});
    const GraphValueId q_rope = rope.outputs[0];
    const GraphValueId k_rope = rope.outputs[1];
    const GraphValueId scores = OnlyOutput(AddPureNode(graph, OpType::kMatMul, layer_index, {q_rope, k_rope}, {specs.scores}));
    const GraphValueId probs = OnlyOutput(AddPureNode(graph, OpType::kSoftmax, layer_index, {scores}, {specs.scores}));
    const GraphValueId attn = OnlyOutput(AddPureNode(graph, OpType::kMatMul, layer_index, {probs, v}, {specs.hidden}));
    const GraphValueId attn_out = OnlyOutput(AddWeightedNode(graph,
                                                             OpType::kLinear,
                                                             layer_index,
                                                             {attn},
                                                             WeightedNodeSpecs{.weight = WeightTensor(layer.attn.o_proj), .output = specs.hidden},
                                                             Bind(ModelWeightRole::kAttentionO, layer_index)));
    const GraphValueId post_attn = OnlyOutput(AddPureNode(graph, OpType::kAdd, layer_index, {hidden, attn_out}, {specs.hidden}));
    const GraphValueId mlp_normed = OnlyOutput(AddRmsNormNode(graph,
                                                              layer_index,
                                                              post_attn,
                                                              WeightedNodeSpecs{.weight = WeightTensor(layer.norm.post_attn_rmsnorm), .output = specs.hidden},
                                                              rms_norm_eps,
                                                              Bind(ModelWeightRole::kPostAttentionNorm, layer_index)));
    const GraphValueId gate = OnlyOutput(AddWeightedNode(graph,
                                                         OpType::kLinear,
                                                         layer_index,
                                                         {mlp_normed},
                                                         WeightedNodeSpecs{.weight = WeightTensor(layer.mlp.gate_proj), .output = specs.intermediate},
                                                         Bind(ModelWeightRole::kMlpGate, layer_index)));
    const GraphValueId up = OnlyOutput(AddWeightedNode(graph,
                                                       OpType::kLinear,
                                                       layer_index,
                                                       {mlp_normed},
                                                       WeightedNodeSpecs{.weight = WeightTensor(layer.mlp.up_proj), .output = specs.intermediate},
                                                       Bind(ModelWeightRole::kMlpUp, layer_index)));
    const GraphValueId mlp_act = OnlyOutput(AddPureNode(graph, OpType::kSiluMul, layer_index, {gate, up}, {specs.intermediate}));
    const GraphValueId mlp_out = OnlyOutput(AddWeightedNode(graph,
                                                            OpType::kLinear,
                                                            layer_index,
                                                            {mlp_act},
                                                            WeightedNodeSpecs{.weight = WeightTensor(layer.mlp.down_proj), .output = specs.hidden},
                                                            Bind(ModelWeightRole::kMlpDown, layer_index)));
    const GraphValueId hidden_out = OnlyOutput(AddPureNode(graph, OpType::kAdd, layer_index, {post_attn, mlp_out}, {specs.hidden}));

    AM_CHECK(graph.GetNodes().size() - layer_begin == kDenseLlamaLayerNodeCount,
             "Dense Llama layer node count changed unexpectedly");
    return hidden_out;
}

Status ValidateInputs(const HfModelConfig& config, const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(config));
    return HfModelValidator::ValidateResolvedModel(config, weights);
}

}// namespace

StatusOr<ModelGraph> ModelGraphBuilder::BuildLlamaDense(const HfModelConfig& config,
                                                        const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(ValidateInputs(config, weights));
    const DataType act_dtype = !config.weight_dtype_hint.IsUndefined() ? config.weight_dtype_hint
                                                                       : DataType::Float32();
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const int64_t hidden_size = config.hidden_size;
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim
                                                  : config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden_size = config.num_key_value_heads * head_dim;

    const TensorSpec token_ids = SymbolicTensorSpec(DataType::Int(64), {seq_len});
    const TensorSpec hidden = ActivationTensor(act_dtype, seq_len, hidden_size);
    const TensorSpec kv_hidden = ActivationTensor(act_dtype, seq_len, kv_hidden_size);
    const TensorSpec intermediate = ActivationTensor(act_dtype, seq_len, config.intermediate_size);
    const TensorSpec scores = SymbolicTensorSpec(act_dtype, {seq_len, seq_len});
    const TensorSpec logits = ActivationTensor(act_dtype, seq_len, config.vocab_size);
    const auto rms_norm_eps = static_cast<float>(config.rms_norm_eps);

    ModelGraph graph(config);

    const GraphValueId input_tokens = graph.AddInput(token_ids, "token_ids");
    const ModelGraph::AddedNode embedding = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {input_tokens, graph.AddWeight(WeightTensor(weights.embed_tokens), Bind(ModelWeightRole::kTokenEmbedding))},
            {ModelGraph::NodeOutputDecl{.spec = hidden, .payload = ActivationValue{}}},
            EmbeddingOp::Params{});
    GraphValueId hidden_value = OnlyOutput(embedding);

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const DecoderLayerRawWeights& layer = weights.layers[layer_index];
        hidden_value = AppendDenseLlamaLayerNodes(graph,
                                                  layer_index,
                                                  layer,
                                                  hidden_value,
                                                  LayerTensorSpecs{
                                                          .hidden = hidden,
                                                          .kv_hidden = kv_hidden,
                                                          .intermediate = intermediate,
                                                          .scores = scores,
                                                  },
                                                  rms_norm_eps);
    }

    const GraphValueId final_hidden = OnlyOutput(AddRmsNormNode(graph,
                                                                std::nullopt,
                                                                hidden_value,
                                                                WeightedNodeSpecs{.weight = WeightTensor(weights.final_norm), .output = hidden},
                                                                rms_norm_eps,
                                                                Bind(ModelWeightRole::kFinalNorm)));

    const RawWeightView& lm_head_weight = weights.lm_head.has_value() ? *weights.lm_head : weights.embed_tokens;
    const GraphValueId logits_value = OnlyOutput(AddWeightedNode(graph,
                                                                 OpType::kLinear,
                                                                 std::nullopt,
                                                                 {final_hidden},
                                                                 WeightedNodeSpecs{.weight = WeightTensor(lm_head_weight), .output = logits},
                                                                 Bind(ModelWeightRole::kLmHead)));
    const GraphValueId output_tokens = OnlyOutput(AddPureNode(graph,
                                                              OpType::kArgmax,
                                                              std::nullopt,
                                                              {logits_value},
                                                              {token_ids}));
    graph.MarkOutput(output_tokens, "output_token_ids");

    AM_RETURN_IF_ERROR(graph.Validate());
    return graph;
}

}// namespace aethermind
