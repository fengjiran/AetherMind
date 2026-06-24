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
    return SymbolicTensorSpec(dtype, {ShapeSymbol::CreateFromValue(num_kv_heads),
                                      seq_len,
                                      ShapeSymbol::CreateFromValue(head_dim)});
}

WeightBinding Bind(WeightRole role, std::optional<uint32_t> decoder_layer_index = std::nullopt) noexcept {
    return WeightBinding{.decoder_layer_index = decoder_layer_index, .role = role};
}

struct LayerTensorSpecs {
    const TensorSpec& hidden_spec;
    const TensorSpec& kv_hidden_spec;
    const TensorSpec& intermediate_spec;
    const TensorSpec& kv_cache_spec;
};

struct LayerInputs {
    GraphValueId hidden;
    GraphValueId kv_cache;
    GraphValueId position_ids;
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
                         {ModelGraph::NodeOutputDesc{.spec = std::move(specs.output), .payload = ActivationValue{}}},
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
    std::vector<ModelGraph::NodeOutputDesc> output_decls;
    output_decls.reserve(outputs.size());
    for (TensorSpec& output: outputs) {
        output_decls.push_back(ModelGraph::NodeOutputDesc{.spec = std::move(output), .payload = output_payload});
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

GraphValueId AppendDenseLlamaLayerNodes(ModelGraph& graph,
                                        uint32_t layer_index,
                                        const DecoderLayerRawWeights& layer_raw_weights,
                                        LayerInputs layer_inputs,
                                        LayerTensorSpecs specs,
                                        float rms_norm_eps,
                                        RoPEParams rope_params,
                                        AttentionParams attention_params) {
    const size_t layer_begin = graph.GetNodes().size();

    const GraphValueId normed = OnlyOneOutput(
            AddRmsNormNode(graph,
                           layer_index,
                           layer_inputs.hidden,
                           WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.norm.input_rmsnorm),
                                             .output = specs.hidden_spec},
                           rms_norm_eps,
                           Bind(WeightRole::kInputNorm, layer_index)));
    const GraphValueId q = OnlyOneOutput(
            AddWeightedNode(graph,
                            OpType::kLinear,
                            layer_index,
                            {normed},
                            WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.attn.q_proj),
                                              .output = specs.hidden_spec},
                            Bind(WeightRole::kAttentionQ, layer_index),
                            LinearParams{}));
    const GraphValueId k = OnlyOneOutput(
            AddWeightedNode(graph,
                            OpType::kLinear,
                            layer_index,
                            {normed},
                            WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.attn.k_proj),
                                              .output = specs.kv_hidden_spec},
                            Bind(WeightRole::kAttentionK, layer_index),
                            LinearParams{}));
    const GraphValueId v = OnlyOneOutput(
            AddWeightedNode(graph,
                            OpType::kLinear,
                            layer_index,
                            {normed},
                            WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.attn.v_proj),
                                              .output = specs.kv_hidden_spec},
                            Bind(WeightRole::kAttentionV, layer_index),
                            LinearParams{}));
    const auto rope = AddPureNode(
            graph,
            OpType::kRoPE,
            layer_index,
            {q, k, layer_inputs.position_ids},
            {specs.hidden_spec, specs.kv_hidden_spec},
            rope_params);
    const GraphValueId q_rope = rope.outputs[0];
    const GraphValueId k_rope = rope.outputs[1];

    StateBinding kv_cache_binding = std::get<StateValue>(
                                            graph.GetValue(layer_inputs.kv_cache).payload)
                                            .binding;
    kv_cache_binding.decoder_layer_index = layer_index;
    const GraphValueId kv_cache_out = OnlyOneOutput(AddPureNode(graph,
                                                                OpType::kKVCacheUpdate,
                                                                layer_index,
                                                                {k_rope, v, layer_inputs.kv_cache},
                                                                {specs.kv_cache_spec},
                                                                KVCacheUpdateParams{},
                                                                StateValue{.binding = kv_cache_binding}));
    const GraphValueId attn = OnlyOneOutput(AddPureNode(graph,
                                                        OpType::kAttention,
                                                        layer_index,
                                                        {q_rope, kv_cache_out},
                                                        {specs.hidden_spec},
                                                        attention_params));
    const GraphValueId attn_out = OnlyOneOutput(AddWeightedNode(graph,
                                                                OpType::kLinear,
                                                                layer_index,
                                                                {attn},
                                                                WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.attn.o_proj), .output = specs.hidden_spec},
                                                                Bind(WeightRole::kAttentionO, layer_index),
                                                                LinearParams{}));
    const GraphValueId post_attn = OnlyOneOutput(AddPureNode(graph, OpType::kAdd, layer_index, {layer_inputs.hidden, attn_out}, {specs.hidden_spec}, AddParams{}));
    const GraphValueId mlp_normed = OnlyOneOutput(AddRmsNormNode(graph,
                                                                 layer_index,
                                                                 post_attn,
                                                                 WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.norm.post_attn_rmsnorm), .output = specs.hidden_spec},
                                                                 rms_norm_eps,
                                                                 Bind(WeightRole::kPostAttentionNorm, layer_index)));
    const GraphValueId gate = OnlyOneOutput(AddWeightedNode(graph,
                                                            OpType::kLinear,
                                                            layer_index,
                                                            {mlp_normed},
                                                            WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.mlp.gate_proj), .output = specs.intermediate_spec},
                                                            Bind(WeightRole::kMlpGate, layer_index),
                                                            LinearParams{}));
    const GraphValueId up = OnlyOneOutput(AddWeightedNode(graph,
                                                          OpType::kLinear,
                                                          layer_index,
                                                          {mlp_normed},
                                                          WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.mlp.up_proj), .output = specs.intermediate_spec},
                                                          Bind(WeightRole::kMlpUp, layer_index),
                                                          LinearParams{}));
    const GraphValueId mlp_act = OnlyOneOutput(AddPureNode(graph, OpType::kSiluMul, layer_index, {gate, up}, {specs.intermediate_spec}, SiluMulParams{}));
    const GraphValueId mlp_out = OnlyOneOutput(AddWeightedNode(graph,
                                                               OpType::kLinear,
                                                               layer_index,
                                                               {mlp_act},
                                                               WeightedNodeSpecs{.weight = WeightTensor(layer_raw_weights.mlp.down_proj), .output = specs.hidden_spec},
                                                               Bind(WeightRole::kMlpDown, layer_index),
                                                               LinearParams{}));
    const GraphValueId hidden_out = OnlyOneOutput(AddPureNode(graph, OpType::kAdd, layer_index, {post_attn, mlp_out}, {specs.hidden_spec}, AddParams{}));

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

    const TensorSpec token_ids_spec = SymbolicTensorSpec(DataType::Int(64), {seq_len});
    const TensorSpec position_ids_spec = SymbolicTensorSpec(DataType::Int(64), {seq_len});
    const TensorSpec hidden_spec = ActivationTensor(act_dtype, seq_len, hidden_size);
    const TensorSpec kv_hidden_spec = ActivationTensor(act_dtype, seq_len, kv_hidden_size);
    const TensorSpec intermediate_spec = ActivationTensor(act_dtype, seq_len, config.intermediate_size);
    const TensorSpec kv_cache_spec = KVCacheTensor(act_dtype, config.num_key_value_heads, seq_len, head_dim);
    const TensorSpec logits_spec = ActivationTensor(act_dtype, seq_len, config.vocab_size);
    const auto rms_norm_eps = static_cast<float>(config.rms_norm_eps);

    ModelGraph graph(config);
    const GraphValueId input_tokens = graph.AddInput(token_ids_spec, "token_ids");
    const GraphValueId position_ids = graph.AddInput(position_ids_spec, "position_ids");
    const auto embedding = AddWeightedNode(
            graph,
            OpType::kEmbedding,
            std::nullopt,
            {input_tokens},
            WeightedNodeSpecs{.weight = WeightTensor(weights.embed_tokens),
                              .output = hidden_spec},
            Bind(WeightRole::kTokenEmbedding),
            EmbeddingParams{});
    GraphValueId hidden_value = OnlyOneOutput(embedding);
    const AttentionParams attention_params{
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
            .head_dim = head_dim,
    };

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const DecoderLayerRawWeights& layer_raw_weights = weights.layers[layer_index];
        const GraphValueId kv_cache_value = graph.AddState(
                kv_cache_spec,
                StateBinding{.logical_id = "kv_cache",
                             .kind = StateKind::kKvCache,
                             .decoder_layer_index = layer_index,
                             .slot = "kv"},
                "kv_cache");
        hidden_value = AppendDenseLlamaLayerNodes(
                graph,
                layer_index,
                layer_raw_weights,
                LayerInputs{.hidden = hidden_value,
                            .kv_cache = kv_cache_value,
                            .position_ids = position_ids},
                LayerTensorSpecs{
                        .hidden_spec = hidden_spec,
                        .kv_hidden_spec = kv_hidden_spec,
                        .intermediate_spec = intermediate_spec,
                        .kv_cache_spec = kv_cache_spec,
                },
                rms_norm_eps,
                MakeRoPEParams(config, head_dim),
                attention_params);
    }

    const GraphValueId final_hidden = OnlyOneOutput(
            AddRmsNormNode(graph,
                           std::nullopt,
                           hidden_value,
                           WeightedNodeSpecs{.weight = WeightTensor(weights.final_norm),
                                             .output = hidden_spec},
                           rms_norm_eps,
                           Bind(WeightRole::kFinalNorm)));

    const RawWeightView& lm_head_weight = weights.lm_head.has_value()
                                                  ? *weights.lm_head
                                                  : weights.embed_tokens;
    const GraphValueId logits_value = OnlyOneOutput(
            AddWeightedNode(graph,
                            OpType::kLinear,
                            std::nullopt,
                            {final_hidden},
                            WeightedNodeSpecs{.weight = WeightTensor(lm_head_weight), .output = logits_spec},
                            Bind(WeightRole::kLmHead),
                            LinearParams{}));
    const GraphValueId output_tokens = OnlyOneOutput(AddPureNode(graph,
                                                                 OpType::kArgmax,
                                                                 std::nullopt,
                                                                 {logits_value},
                                                                 {token_ids_spec},
                                                                 ArgmaxParams{.axis = -1}));
    graph.MarkOutput(output_tokens, "output_token_ids");

    AM_RETURN_IF_ERROR(graph.Validate());
    return graph;
}

}// namespace aethermind
