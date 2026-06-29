#include "aethermind/model/graph/model_graph_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace aethermind {
namespace {

// Attention block: input_layernorm, q_proj, k_proj, v_proj, rotary_emb,
// kv_cache_update, attention, o_proj, post_attention_add.
constexpr size_t kAttentionBlockNodeCount = 9;

// MLP block: post_attention_layernorm, gate_proj, up_proj, act, down_proj,
// mlp_add.
constexpr size_t kMlpBlockNodeCount = 6;

TensorSpec WeightTensorSpec(const RawWeightView& weight) {
    return TensorSpec{.dtype = weight.dtype,
                      .shape = SymbolicShape(IntArrayView(weight.shape))};
}

TensorSpec ActivationTensorSpec(DataType dtype, ShapeSymbol seq_len, int64_t feature_dim) {
    return TensorSpec{.dtype = dtype,
                      .shape = SymbolicShape({seq_len, ShapeSymbol::CreateFromValue(feature_dim)})};
}

TensorSpec KVCacheTensorSpec(DataType dtype, int64_t num_kv_heads,
                             ShapeSymbol cache_len, int64_t head_dim) {
    return TensorSpec{.dtype = dtype,
                      .shape = SymbolicShape({ShapeSymbol::CreateFromValue(num_kv_heads),
                                              cache_len,
                                              ShapeSymbol::CreateFromValue(head_dim)})};
}

std::string LayerPrefix(uint32_t layer) {
    return "layers." + std::to_string(layer) + ".";
}

std::string WeightDebugName(WeightRole role, std::optional<uint32_t> layer) {
    switch (role) {
        case WeightRole::kTokenEmbedding:
            AM_CHECK(!layer.has_value(), "Token embedding weight must not be layer-scoped");
            return "embed_tokens";
        case WeightRole::kInputNorm:
            AM_CHECK(layer.has_value(), "Input norm weight must be layer-scoped");
            return LayerPrefix(*layer) + "input_layernorm";
        case WeightRole::kAttentionQ:
            AM_CHECK(layer.has_value(), "Attention Q weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.q_proj";
        case WeightRole::kAttentionK:
            AM_CHECK(layer.has_value(), "Attention K weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.k_proj";
        case WeightRole::kAttentionV:
            AM_CHECK(layer.has_value(), "Attention V weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.v_proj";
        case WeightRole::kAttentionO:
            AM_CHECK(layer.has_value(), "Attention O weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.o_proj";
        case WeightRole::kMlpGate:
            AM_CHECK(layer.has_value(), "MLP gate weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.gate_proj";
        case WeightRole::kMlpUp:
            AM_CHECK(layer.has_value(), "MLP up weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.up_proj";
        case WeightRole::kMlpDown:
            AM_CHECK(layer.has_value(), "MLP down weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.down_proj";
        case WeightRole::kPostAttentionNorm:
            AM_CHECK(layer.has_value(), "Post-attention norm weight must be layer-scoped");
            return LayerPrefix(*layer) + "post_attention_layernorm";
        case WeightRole::kFinalNorm:
            AM_CHECK(!layer.has_value(), "Final norm weight must not be layer-scoped");
            return "norm";
        case WeightRole::kLmHead:
            AM_CHECK(!layer.has_value(), "LM head weight must not be layer-scoped");
            return "lm_head";
    }
    AM_UNREACHABLE();
}

struct QkvProjectionResult {
    GraphValueId q;
    GraphValueId k;
    GraphValueId v;
};

struct RoPEResult {
    GraphValueId q;
    GraphValueId k;
};

struct KVCacheResult {
    GraphValueId k;
    GraphValueId v;
};

struct AttentionBlockResult {
    GraphValueId hidden;
};

struct AttentionBlockInput {
    GraphValueId hidden;
    GraphValueId position_ids;
    KVCacheResult cache;
};

struct LlamaBuildSpecs {
    TensorSpec token_ids;
    TensorSpec position_ids;
    TensorSpec hidden;
    TensorSpec kv_hidden;
    TensorSpec intermediate;
    TensorSpec kv_cache;
    TensorSpec logits;
};

struct LlamaBuildParams {
    float rms_norm_eps = 0.0F;
    RoPEParams rope;
    AttentionParams attention;
};

struct LlamaBuildContext {
    ModelGraph& graph;
    const ResolvedModelWeights& weights;
    const LlamaBuildSpecs& specs;
    const LlamaBuildParams& params;
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

struct WeightedNodeSpecs {
    TensorSpec weight;
    TensorSpec output;
};

ModelGraph::AddedNode AddWeightedNode(ModelGraph& graph,
                                      OpType op_type,
                                      std::optional<uint32_t> decoder_layer_index,
                                      std::vector<GraphValueId> inputs,
                                      WeightedNodeSpecs specs,
                                      WeightBinding weight,
                                      const OpParams& op_params,
                                      std::string debug_name = "") {
    inputs.push_back(graph.AddWeight(std::move(specs.weight), weight));
    return graph.AddNode(op_type,
                         decoder_layer_index,
                         std::move(inputs),
                         {ModelGraph::NodeOutputDesc{
                                 .spec = std::move(specs.output),
                                 .payload = ActivationValue{}}},
                         op_params,
                         {},
                         std::move(debug_name));
}

ModelGraph::AddedNode AddPureNode(ModelGraph& graph,
                                  OpType op_type,
                                  std::optional<uint32_t> decoder_layer_index,
                                  std::vector<GraphValueId> inputs,
                                  std::vector<TensorSpec> outputs,
                                  const OpParams& op_params,
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

GraphValueId AddLinear(LlamaBuildContext& ctx,
                       std::optional<uint32_t> layer,
                       GraphValueId input,
                       const RawWeightView& weight,
                       TensorSpec output,
                       WeightRole role) {
    return OnlyOneOutput(AddWeightedNode(
            ctx.graph,
            OpType::kLinear,
            layer,
            {input},
            WeightedNodeSpecs{.weight = WeightTensorSpec(weight), .output = std::move(output)},
            WeightBinding{.decoder_layer_index = layer, .role = role},
            LinearParams{},
            WeightDebugName(role, layer)));
}

GraphValueId AddRmsNorm(LlamaBuildContext& ctx,
                        std::optional<uint32_t> layer,
                        GraphValueId input,
                        const RawWeightView& weight,
                        WeightRole role) {
    return OnlyOneOutput(AddWeightedNode(
            ctx.graph,
            OpType::kRmsNorm,
            layer,
            {input},
            WeightedNodeSpecs{.weight = WeightTensorSpec(weight), .output = ctx.specs.hidden},
            WeightBinding{.decoder_layer_index = layer, .role = role},
            RmsNormParams{.eps = ctx.params.rms_norm_eps},
            WeightDebugName(role, layer)));
}

GraphValueId AddEmbedding(LlamaBuildContext& ctx, GraphValueId input) {
    return OnlyOneOutput(AddWeightedNode(
            ctx.graph,
            OpType::kEmbedding,
            std::nullopt,
            {input},
            WeightedNodeSpecs{.weight = WeightTensorSpec(ctx.weights.embed_tokens), .output = ctx.specs.hidden},
            WeightBinding{.role = WeightRole::kTokenEmbedding},
            EmbeddingParams{},
            WeightDebugName(WeightRole::kTokenEmbedding, std::nullopt)));
}

QkvProjectionResult AddQkvProjections(LlamaBuildContext& ctx,
                                      uint32_t layer,
                                      GraphValueId input,
                                      const AttnRawWeights& attn_weights) {
    return QkvProjectionResult{
            .q = AddLinear(ctx, layer, input, attn_weights.q_proj, ctx.specs.hidden, WeightRole::kAttentionQ),
            .k = AddLinear(ctx, layer, input, attn_weights.k_proj, ctx.specs.kv_hidden, WeightRole::kAttentionK),
            .v = AddLinear(ctx, layer, input, attn_weights.v_proj, ctx.specs.kv_hidden, WeightRole::kAttentionV),
    };
}

RoPEResult AddRoPE(LlamaBuildContext& ctx,
                   uint32_t layer,
                   GraphValueId q,
                   GraphValueId k,
                   GraphValueId position_ids) {
    const ModelGraph::AddedNode node = AddPureNode(
            ctx.graph,
            OpType::kRoPE,
            layer,
            {q, k, position_ids},
            {ctx.specs.hidden, ctx.specs.kv_hidden},
            ctx.params.rope,
            ActivationValue{},
            LayerPrefix(layer) + "self_attn.rotary_emb");
    return RoPEResult{.q = node.outputs[0], .k = node.outputs[1]};
}

KVCacheResult AddKVCacheUpdate(LlamaBuildContext& ctx,
                               uint32_t layer,
                               GraphValueId k_new,
                               GraphValueId v_new,
                               KVCacheResult cache_in) {
    const GraphValue& k_cache_value = ctx.graph.GetValue(cache_in.k);
    const auto* k_cache_state = std::get_if<StateValue>(&k_cache_value.payload);
    AM_CHECK(k_cache_state != nullptr, "K cache input must be a StateValue");
    const GraphValue& v_cache_value = ctx.graph.GetValue(cache_in.v);
    const auto* v_cache_state = std::get_if<StateValue>(&v_cache_value.payload);
    AM_CHECK(v_cache_state != nullptr, "V cache input must be a StateValue");

    const ModelGraph::AddedNode node = ctx.graph.AddNode(
            OpType::kKVCacheUpdate,
            layer,
            {k_new, v_new, cache_in.k, cache_in.v},
            {ModelGraph::NodeOutputDesc{.spec = ctx.specs.kv_cache,
                                        .payload = StateValue{.binding = k_cache_state->binding}},// NOLINT
             ModelGraph::NodeOutputDesc{.spec = ctx.specs.kv_cache,
                                        .payload = StateValue{.binding = v_cache_state->binding}}},// NOLINT
            KVCacheUpdateParams{},
            {},
            LayerPrefix(layer) + "self_attn.kv_cache_update");
    return KVCacheResult{.k = node.outputs[0], .v = node.outputs[1]};
}

GraphValueId AddAttention(LlamaBuildContext& ctx,
                          uint32_t layer,
                          GraphValueId q,
                          KVCacheResult cache) {
    return OnlyOneOutput(AddPureNode(
            ctx.graph,
            OpType::kAttention,
            layer,
            {q, cache.k, cache.v},
            {ctx.specs.hidden},
            ctx.params.attention,
            ActivationValue{},
            LayerPrefix(layer) + "self_attn.attention"));
}

GraphValueId AddResidualAdd(LlamaBuildContext& ctx,
                            std::optional<uint32_t> layer,
                            GraphValueId lhs,
                            GraphValueId rhs,
                            std::string debug_name) {
    return OnlyOneOutput(AddPureNode(
            ctx.graph,
            OpType::kAdd,
            layer,
            {lhs, rhs},
            {ctx.specs.hidden},
            AddParams{},
            ActivationValue{},
            std::move(debug_name)));
}

GraphValueId AddSiluMul(LlamaBuildContext& ctx,
                        uint32_t layer,
                        GraphValueId gate,
                        GraphValueId up) {
    return OnlyOneOutput(AddPureNode(
            ctx.graph,
            OpType::kSiluMul,
            layer,
            {gate, up},
            {ctx.specs.intermediate},
            SiluMulParams{},
            ActivationValue{},
            LayerPrefix(layer) + "mlp.act"));
}

AttentionBlockResult BuildLlamaAttentionBlock(LlamaBuildContext& ctx,
                                              uint32_t layer,
                                              AttentionBlockInput input) {
    const size_t block_begin = ctx.graph.GetNodes().size();

    const GraphValueId normed = AddRmsNorm(ctx, layer, input.hidden,
                                           ctx.weights.layers[layer].norm.input_rmsnorm,
                                           WeightRole::kInputNorm);
    const QkvProjectionResult qkv = AddQkvProjections(ctx, layer, normed,
                                                      ctx.weights.layers[layer].attn);
    const RoPEResult rope = AddRoPE(ctx, layer, qkv.q, qkv.k, input.position_ids);
    const KVCacheResult cache_out = AddKVCacheUpdate(ctx, layer, rope.k, qkv.v, input.cache);
    const GraphValueId attn = AddAttention(ctx, layer, rope.q, cache_out);
    const GraphValueId o_proj = AddLinear(ctx, layer, attn,
                                          ctx.weights.layers[layer].attn.o_proj,
                                          ctx.specs.hidden, WeightRole::kAttentionO);
    const GraphValueId residual = AddResidualAdd(ctx, layer, input.hidden, o_proj,
                                                 LayerPrefix(layer) + "post_attention_add");

    AM_CHECK(ctx.graph.GetNodes().size() - block_begin == kAttentionBlockNodeCount,
             "Attention block node count changed unexpectedly");
    return AttentionBlockResult{.hidden = residual};
}

GraphValueId BuildLlamaMlpBlock(LlamaBuildContext& ctx,
                                uint32_t layer,
                                GraphValueId hidden) {
    const size_t block_begin = ctx.graph.GetNodes().size();

    const GraphValueId normed = AddRmsNorm(ctx, layer, hidden,
                                           ctx.weights.layers[layer].norm.post_attn_rmsnorm,
                                           WeightRole::kPostAttentionNorm);
    const GraphValueId gate = AddLinear(ctx, layer, normed,
                                        ctx.weights.layers[layer].mlp.gate_proj,
                                        ctx.specs.intermediate, WeightRole::kMlpGate);
    const GraphValueId up = AddLinear(ctx, layer, normed,
                                      ctx.weights.layers[layer].mlp.up_proj,
                                      ctx.specs.intermediate, WeightRole::kMlpUp);
    const GraphValueId act = AddSiluMul(ctx, layer, gate, up);
    const GraphValueId down = AddLinear(ctx, layer, act,
                                        ctx.weights.layers[layer].mlp.down_proj,
                                        ctx.specs.hidden, WeightRole::kMlpDown);
    const GraphValueId residual = AddResidualAdd(ctx, layer, hidden, down,
                                                 LayerPrefix(layer) + "mlp_add");

    AM_CHECK(ctx.graph.GetNodes().size() - block_begin == kMlpBlockNodeCount,
             "MLP block node count changed unexpectedly");
    return residual;
}

GraphValueId BuildLlamaDecoderLayer(LlamaBuildContext& ctx,
                                    uint32_t layer,
                                    GraphValueId hidden,
                                    GraphValueId position_ids,
                                    KVCacheResult cache_in) {
    const AttentionBlockResult attn = BuildLlamaAttentionBlock(
            ctx,
            layer,
            AttentionBlockInput{.hidden = hidden,
                                .position_ids = position_ids,
                                .cache = cache_in});
    return BuildLlamaMlpBlock(ctx, layer, attn.hidden);
}

Status ValidateInputs(const HfModelConfig& config, const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(config));
    return HfModelValidator::ValidateResolvedModel(config, weights);
}

}// namespace

StatusOr<ModelGraph> ModelGraphBuilder::BuildLlamaDense(const HfModelConfig& config,
                                                        const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(ValidateInputs(config, weights));
    const DataType act_dtype = config.weight_dtype_hint.IsUndefined()
                                       ? DataType::Float32()
                                       : config.weight_dtype_hint;
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol kv_len = ShapeSymbol::Create();
    const int64_t hidden_size = config.hidden_size;
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim
                                                  : config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden_size = config.num_key_value_heads * head_dim;

    const LlamaBuildSpecs specs{
            .token_ids = TensorSpec{.dtype = DataType::Int(64), .shape = SymbolicShape({seq_len})},
            .position_ids = TensorSpec{.dtype = DataType::Int(64), .shape = SymbolicShape({seq_len})},
            .hidden = ActivationTensorSpec(act_dtype, seq_len, hidden_size),
            .kv_hidden = ActivationTensorSpec(act_dtype, seq_len, kv_hidden_size),
            .intermediate = ActivationTensorSpec(act_dtype, seq_len, config.intermediate_size),
            .kv_cache = KVCacheTensorSpec(act_dtype, config.num_key_value_heads, kv_len, head_dim),
            .logits = ActivationTensorSpec(act_dtype, seq_len, config.vocab_size),
    };
    const LlamaBuildParams params{
            .rms_norm_eps = static_cast<float>(config.rms_norm_eps),
            .rope = MakeRoPEParams(config, head_dim),
            .attention = AttentionParams{
                    .num_attention_heads = config.num_attention_heads,
                    .num_key_value_heads = config.num_key_value_heads,
                    .head_dim = head_dim,
            },
    };

    ModelGraph graph(config);
    LlamaBuildContext ctx{
            .graph = graph,
            .weights = weights,
            .specs = specs,
            .params = params,
    };

    const GraphValueId input_tokens = ctx.graph.AddInput(specs.token_ids, "token_ids");
    const GraphValueId position_ids = ctx.graph.AddInput(specs.position_ids, "position_ids");

    GraphValueId hidden = AddEmbedding(ctx, input_tokens);

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const GraphValueId k_cache = ctx.graph.AddState(
                specs.kv_cache,
                KVCacheStateBinding{.decoder_layer_index = layer_index, .slot = KVCacheSlot::kKey},
                LayerPrefix(layer_index) + "self_attn.k_cache");
        const GraphValueId v_cache = ctx.graph.AddState(
                specs.kv_cache,
                KVCacheStateBinding{.decoder_layer_index = layer_index, .slot = KVCacheSlot::kValue},
                LayerPrefix(layer_index) + "self_attn.v_cache");
        hidden = BuildLlamaDecoderLayer(
                ctx, layer_index, hidden, position_ids,
                KVCacheResult{.k = k_cache, .v = v_cache});
    }

    const GraphValueId final_hidden = AddRmsNorm(ctx, std::nullopt, hidden,
                                                 ctx.weights.final_norm, WeightRole::kFinalNorm);
    const RawWeightView& lm_head_weight = ctx.weights.lm_head.has_value()
                                                  ? *ctx.weights.lm_head
                                                  : ctx.weights.embed_tokens;
    const GraphValueId logits = AddLinear(ctx, std::nullopt, final_hidden, lm_head_weight,
                                          specs.logits, WeightRole::kLmHead);
    const GraphValueId output_tokens = OnlyOneOutput(AddPureNode(
            ctx.graph,
            OpType::kArgmax,
            std::nullopt,
            {logits},
            {specs.token_ids},
            ArgmaxParams{.axis = -1},
            ActivationValue{},
            "argmax"));
    ctx.graph.MarkOutput(output_tokens, "output_token_ids");

    AM_RETURN_IF_ERROR(ctx.graph.Validate());
    return graph;
}

}// namespace aethermind
