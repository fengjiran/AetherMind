#include "aethermind/model/graph/model_graph_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace aethermind {
namespace {

// Decoder-only dense Llama layer nodes, in execution order:
// input RMSNorm, Q/K/V projection, RoPE, KV cache update, attention,
// output projection, residual Add, post-attention RMSNorm,
// MLP gate/up projection, SiLU multiply, MLP down projection, residual Add.
constexpr size_t kDenseLlamaLayerNodeCount = 15;

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

TensorSpec KVCacheTensor(DataType dtype, int64_t num_kv_heads, ShapeSymbol seq_len, int64_t head_dim) {
    return SymbolicTensorSpec(dtype, {ShapeSymbol::CreateFromValue(num_kv_heads), seq_len, ShapeSymbol::CreateFromValue(head_dim)});
}

WeightBinding Bind(WeightRole role, std::optional<uint32_t> decoder_layer_index = std::nullopt) noexcept {
    return WeightBinding{.decoder_layer_index = decoder_layer_index, .role = role};
}

struct LayerTensorSpecs {
    const TensorSpec& hidden;
    const TensorSpec& kv_hidden;
    const TensorSpec& intermediate;
    const TensorSpec& kv_cache;
};

struct LayerOutputs {
    GraphValueId hidden;
    GraphValueId kv_cache;
};

struct LayerInputs {
    GraphValueId hidden;
    GraphValueId kv_cache;
};

struct WeightedNodeSpecs {
    TensorSpec weight;
    TensorSpec output;
};

RoPEParams MakeRoPEParams(const HfModelConfig& config, int64_t head_dim) {
    return RoPEParams{
            .head_dim = head_dim,
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .max_position_embeddings = config.max_position_embeddings,
            .theta = config.rope.theta,
            .scaling_factor = config.rope.scaling_factor,
            .scaling_type = config.rope.scaling_type,
    };
}

GraphValueId OnlyOneOutput(const ModelGraph::AddedNode& added_node) {
    AM_CHECK(added_node.outputs.size() == 1U, "Expected node to have exactly one output");
    return added_node.outputs.front();
}

ModelGraph::AddedNode AddWeightedNode(ModelGraph& graph,
                                      OpType op_type,
                                      std::optional<uint32_t> decoder_layer_index,
                                      std::vector<GraphValueId> inputs,
                                      WeightedNodeSpecs specs,
                                      WeightBinding weight,
                                      OpParams op_params,
                                      std::string debug_name = "") {
    inputs.push_back(graph.AddWeight(std::move(specs.weight), weight));
    return graph.AddNode(op_type,
                         decoder_layer_index,
                         std::move(inputs),
                         {ModelGraph::NodeOutputDecl{.spec = std::move(specs.output), .payload = ActivationValue{}}},
                         op_params,
                         {},
                         std::move(debug_name));
}

ModelGraph::AddedNode AddPureNode(ModelGraph& graph,
                                  OpType op_type,
                                  std::optional<uint32_t> decoder_layer_index,
                                  std::vector<GraphValueId> inputs,
                                  std::vector<TensorSpec> outputs,
                                  OpParams op_params,
                                  const GraphValuePayload& output_payload = ActivationValue{},
                                  std::string debug_name = "") {
    std::vector<ModelGraph::NodeOutputDecl> output_decls;
    output_decls.reserve(outputs.size());
    for (TensorSpec& output: outputs) {
        output_decls.push_back(ModelGraph::NodeOutputDecl{.spec = std::move(output), .payload = output_payload});
    }
    return graph.AddNode(op_type,
                         decoder_layer_index,
                         std::move(inputs),
                         std::move(output_decls),
                         op_params,
                         {},
                         std::move(debug_name));
}

ModelGraph::AddedNode AddRmsNormNode(ModelGraph& graph,
                                     std::optional<uint32_t> decoder_layer_index,
                                     GraphValueId input,
                                     WeightedNodeSpecs specs,
                                     float eps,
                                     WeightBinding weight) {
    return AddWeightedNode(graph,
                           OpType::kRmsNorm,
                           decoder_layer_index,
                           {input},
                           std::move(specs),
                           weight,
                           RmsNormParams{.eps = eps});
}

LayerOutputs AppendDenseLlamaLayerNodes(ModelGraph& graph,
                                        uint32_t layer_index,
                                        const DecoderLayerRawWeights& layer,
                                        LayerInputs layer_inputs,
                                        LayerTensorSpecs specs,
                                        float rms_norm_eps,
                                        RoPEParams rope_params,
                                        AttentionParams attention_params) {
    const size_t layer_begin = graph.GetNodes().size();

    const GraphValueId normed = OnlyOneOutput(AddRmsNormNode(graph,
                                                             layer_index,
                                                             layer_inputs.hidden,
                                                             WeightedNodeSpecs{.weight = WeightTensor(layer.norm.input_rmsnorm), .output = specs.hidden},
                                                             rms_norm_eps,
                                                             Bind(WeightRole::kInputNorm, layer_index)));
    const GraphValueId q = OnlyOneOutput(AddWeightedNode(graph,
                                                         OpType::kLinear,
                                                         layer_index,
                                                         {normed},
                                                         WeightedNodeSpecs{.weight = WeightTensor(layer.attn.q_proj), .output = specs.hidden},
                                                         Bind(WeightRole::kAttentionQ, layer_index),
                                                         LinearParams{}));
    const GraphValueId k = OnlyOneOutput(AddWeightedNode(graph,
                                                         OpType::kLinear,
                                                         layer_index,
                                                         {normed},
                                                         WeightedNodeSpecs{.weight = WeightTensor(layer.attn.k_proj), .output = specs.kv_hidden},
                                                         Bind(WeightRole::kAttentionK, layer_index),
                                                         LinearParams{}));
    const GraphValueId v = OnlyOneOutput(AddWeightedNode(graph,
                                                         OpType::kLinear,
                                                         layer_index,
                                                         {normed},
                                                         WeightedNodeSpecs{.weight = WeightTensor(layer.attn.v_proj), .output = specs.kv_hidden},
                                                         Bind(WeightRole::kAttentionV, layer_index),
                                                         LinearParams{}));
    const ModelGraph::AddedNode rope = AddPureNode(graph, OpType::kRoPE, layer_index, {q, k}, {specs.hidden, specs.kv_hidden}, rope_params);
    const GraphValueId q_rope = rope.outputs[0];
    const GraphValueId k_rope = rope.outputs[1];

    StateBinding kv_cache_binding = std::get<StateValue>(graph.GetValue(layer_inputs.kv_cache).payload).binding;
    kv_cache_binding.decoder_layer_index = layer_index;
    const GraphValueId kv_cache_out = OnlyOneOutput(AddPureNode(graph,
                                                                OpType::kKVCacheUpdate,
                                                                layer_index,
                                                                {k_rope, v, layer_inputs.kv_cache},
                                                                {specs.kv_cache},
                                                                KVCacheUpdateParams{},
                                                                StateValue{.binding = kv_cache_binding}));
    const GraphValueId attn = OnlyOneOutput(AddPureNode(graph,
                                                        OpType::kAttention,
                                                        layer_index,
                                                        {q_rope, kv_cache_out},
                                                        {specs.hidden},
                                                        attention_params));
    const GraphValueId attn_out = OnlyOneOutput(AddWeightedNode(graph,
                                                                OpType::kLinear,
                                                                layer_index,
                                                                {attn},
                                                                WeightedNodeSpecs{.weight = WeightTensor(layer.attn.o_proj), .output = specs.hidden},
                                                                Bind(WeightRole::kAttentionO, layer_index),
                                                                LinearParams{}));
    const GraphValueId post_attn = OnlyOneOutput(AddPureNode(graph, OpType::kAdd, layer_index, {layer_inputs.hidden, attn_out}, {specs.hidden}, AddParams{}));
    const GraphValueId mlp_normed = OnlyOneOutput(AddRmsNormNode(graph,
                                                                 layer_index,
                                                                 post_attn,
                                                                 WeightedNodeSpecs{.weight = WeightTensor(layer.norm.post_attn_rmsnorm), .output = specs.hidden},
                                                                 rms_norm_eps,
                                                                 Bind(WeightRole::kPostAttentionNorm, layer_index)));
    const GraphValueId gate = OnlyOneOutput(AddWeightedNode(graph,
                                                            OpType::kLinear,
                                                            layer_index,
                                                            {mlp_normed},
                                                            WeightedNodeSpecs{.weight = WeightTensor(layer.mlp.gate_proj), .output = specs.intermediate},
                                                            Bind(WeightRole::kMlpGate, layer_index),
                                                            LinearParams{}));
    const GraphValueId up = OnlyOneOutput(AddWeightedNode(graph,
                                                          OpType::kLinear,
                                                          layer_index,
                                                          {mlp_normed},
                                                          WeightedNodeSpecs{.weight = WeightTensor(layer.mlp.up_proj), .output = specs.intermediate},
                                                          Bind(WeightRole::kMlpUp, layer_index),
                                                          LinearParams{}));
    const GraphValueId mlp_act = OnlyOneOutput(AddPureNode(graph, OpType::kSiluMul, layer_index, {gate, up}, {specs.intermediate}, SiluMulParams{}));
    const GraphValueId mlp_out = OnlyOneOutput(AddWeightedNode(graph,
                                                               OpType::kLinear,
                                                               layer_index,
                                                               {mlp_act},
                                                               WeightedNodeSpecs{.weight = WeightTensor(layer.mlp.down_proj), .output = specs.hidden},
                                                               Bind(WeightRole::kMlpDown, layer_index),
                                                               LinearParams{}));
    const GraphValueId hidden_out = OnlyOneOutput(AddPureNode(graph, OpType::kAdd, layer_index, {post_attn, mlp_out}, {specs.hidden}, AddParams{}));

    AM_CHECK(graph.GetNodes().size() - layer_begin == kDenseLlamaLayerNodeCount,
             "Dense Llama layer node count changed unexpectedly");
    return LayerOutputs{.hidden = hidden_out, .kv_cache = kv_cache_out};
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
    const TensorSpec kv_cache = KVCacheTensor(act_dtype, config.num_key_value_heads, seq_len, head_dim);
    const TensorSpec logits = ActivationTensor(act_dtype, seq_len, config.vocab_size);
    const auto rms_norm_eps = static_cast<float>(config.rms_norm_eps);

    ModelGraph graph(config);

    const GraphValueId input_tokens = graph.AddInput(token_ids, "token_ids");
    const ModelGraph::AddedNode embedding = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {input_tokens, graph.AddWeight(WeightTensor(weights.embed_tokens), Bind(WeightRole::kTokenEmbedding))},
            {ModelGraph::NodeOutputDecl{.spec = hidden, .payload = ActivationValue{}}},
            EmbeddingParams{});
    GraphValueId hidden_value = OnlyOneOutput(embedding);
    GraphValueId kv_cache_value = graph.AddState(
            kv_cache,
            StateBinding{
                    .logical_id = "kv_cache",
                    .kind = StateKind::kKvCache,
                    .slot = "kv"},
            "kv_cache");

    const AttentionParams attention_params{
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .head_dim = head_dim,
    };

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const DecoderLayerRawWeights& layer = weights.layers[layer_index];
        const LayerOutputs layer_outputs = AppendDenseLlamaLayerNodes(graph,
                                                                      layer_index,
                                                                      layer,
                                                                      LayerInputs{.hidden = hidden_value, .kv_cache = kv_cache_value},
                                                                      LayerTensorSpecs{
                                                                              .hidden = hidden,
                                                                              .kv_hidden = kv_hidden,
                                                                              .intermediate = intermediate,
                                                                              .kv_cache = kv_cache,
                                                                      },
                                                                      rms_norm_eps,
                                                                      MakeRoPEParams(config, head_dim),
                                                                      attention_params);
        hidden_value = layer_outputs.hidden;
        kv_cache_value = layer_outputs.kv_cache;
    }

    const GraphValueId final_hidden = OnlyOneOutput(AddRmsNormNode(graph,
                                                                   std::nullopt,
                                                                   hidden_value,
                                                                   WeightedNodeSpecs{.weight = WeightTensor(weights.final_norm), .output = hidden},
                                                                   rms_norm_eps,
                                                                   Bind(WeightRole::kFinalNorm)));

    const RawWeightView& lm_head_weight = weights.lm_head.has_value() ? *weights.lm_head : weights.embed_tokens;
    const GraphValueId logits_value = OnlyOneOutput(AddWeightedNode(graph,
                                                                    OpType::kLinear,
                                                                    std::nullopt,
                                                                    {final_hidden},
                                                                    WeightedNodeSpecs{.weight = WeightTensor(lm_head_weight), .output = logits},
                                                                    Bind(WeightRole::kLmHead),
                                                                    LinearParams{}));
    const GraphValueId output_tokens = OnlyOneOutput(AddPureNode(graph,
                                                                 OpType::kArgmax,
                                                                 std::nullopt,
                                                                 {logits_value},
                                                                 {token_ids},
                                                                 ArgmaxParams{.axis = -1}));
    graph.MarkOutput(output_tokens, "output_token_ids");

    AM_RETURN_IF_ERROR(graph.Validate());
    return graph;
}

}// namespace aethermind
