#include "aethermind/model/graph/graph_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/graph/graph_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace aethermind {
namespace {

// Attention block: input_layernorm, q_proj, k_proj, v_proj, rotary_emb,
// kv_cache_update, attention, o_proj, post_attention_add.
constexpr size_t kAttentionBlockNodeCount = 9;

// MLP block: post_attention_layernorm, gate_proj, up_proj, act, down_proj,
// mlp_add.
constexpr size_t kMlpBlockNodeCount = 6;

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

std::string WeightDebugName(TransformerWeightRole role, std::optional<uint32_t> layer) {
    switch (role) {
        case TransformerWeightRole::kTokenEmbedding:
            AM_CHECK(!layer.has_value(), "Token embedding weight must not be layer-scoped");
            return "embed_tokens";
        case TransformerWeightRole::kInputNorm:
            AM_CHECK(layer.has_value(), "Input norm weight must be layer-scoped");
            return LayerPrefix(*layer) + "input_layernorm";
        case TransformerWeightRole::kAttentionQ:
            AM_CHECK(layer.has_value(), "Attention Q weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.q_proj";
        case TransformerWeightRole::kAttentionK:
            AM_CHECK(layer.has_value(), "Attention K weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.k_proj";
        case TransformerWeightRole::kAttentionV:
            AM_CHECK(layer.has_value(), "Attention V weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.v_proj";
        case TransformerWeightRole::kAttentionO:
            AM_CHECK(layer.has_value(), "Attention O weight must be layer-scoped");
            return LayerPrefix(*layer) + "self_attn.o_proj";
        case TransformerWeightRole::kMlpGate:
            AM_CHECK(layer.has_value(), "MLP gate weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.gate_proj";
        case TransformerWeightRole::kMlpUp:
            AM_CHECK(layer.has_value(), "MLP up weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.up_proj";
        case TransformerWeightRole::kMlpDown:
            AM_CHECK(layer.has_value(), "MLP down weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.down_proj";
        case TransformerWeightRole::kPostAttentionNorm:
            AM_CHECK(layer.has_value(), "Post-attention norm weight must be layer-scoped");
            return LayerPrefix(*layer) + "post_attention_layernorm";
        case TransformerWeightRole::kFinalNorm:
            AM_CHECK(!layer.has_value(), "Final norm weight must not be layer-scoped");
            return "norm";
        case TransformerWeightRole::kLmHead:
            AM_CHECK(!layer.has_value(), "LM head weight must not be layer-scoped");
            return "lm_head";
        case TransformerWeightRole::kMoERouter:
            AM_CHECK(layer.has_value(), "MoE router weight must be layer-scoped");
            return LayerPrefix(*layer) + "mlp.router";
    }
    AM_UNREACHABLE();
}

WeightBinding MakeTransformerWeightBinding(std::optional<uint32_t> layer,
                                           TransformerWeightRole role) {
    return WeightBinding{.slot = SlotForTransformerRole(role),
                         .decoder_layer_index = layer,
                         .semantic_role = role};
}

struct AttentionBlockResult {
    GraphValueId hidden;
};

struct AttentionBlockInput {
    GraphValueId hidden;
    GraphValueId position_ids;
    KVCachePair cache;
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

KVCachePair AddLlamaKVCacheUpdate(LlamaBuildContext& ctx,
                                    uint32_t layer,
                                    GraphValueId k_new,
                                    GraphValueId v_new,
                                    KVCachePair cache_in) {
    const GraphValue& k_cache_value = ctx.graph.GetValue(cache_in.k);
    const auto* k_cache_state = std::get_if<StateValue>(&k_cache_value.payload);
    AM_CHECK(k_cache_state != nullptr, "K cache input must be a StateValue");
    const GraphValue& v_cache_value = ctx.graph.GetValue(cache_in.v);
    const auto* v_cache_state = std::get_if<StateValue>(&v_cache_value.payload);
    AM_CHECK(v_cache_state != nullptr, "V cache input must be a StateValue");

    return AddKVCacheUpdate(
            ctx.graph,
            layer,
            k_new,
            v_new,
            cache_in.k,
            cache_in.v,
            ctx.specs.kv_cache,
            ctx.specs.kv_cache,
            k_cache_state->binding,// NOLINT
            v_cache_state->binding,// NOLINT
            LayerPrefix(layer) + "self_attn.kv_cache_update");
}

AttentionBlockResult BuildLlamaAttentionBlock(LlamaBuildContext& ctx,
                                              uint32_t layer,
                                              AttentionBlockInput input) {
    const size_t block_begin = ctx.graph.GetNodes().size();

    const RawWeightView& input_norm_weight = ctx.weights.layers[layer].norm.input_rmsnorm;
    const GraphValueId normed = AddRmsNorm(ctx.graph,
                                           input.hidden,
                                           input_norm_weight.dtype,
                                           MakeTransformerWeightBinding(layer, TransformerWeightRole::kInputNorm),
                                           ctx.params.rms_norm_eps,
                                           WeightDebugName(TransformerWeightRole::kInputNorm, layer));
    const RawWeightView& q_proj_weight = ctx.weights.layers[layer].attn.q_proj;
    const GraphValueId q = AddLinear(ctx.graph,
                                     normed,
                                     q_proj_weight.shape[0],
                                     q_proj_weight.dtype,
                                     MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionQ),
                                     WeightDebugName(TransformerWeightRole::kAttentionQ, layer));
    const RawWeightView& k_proj_weight = ctx.weights.layers[layer].attn.k_proj;
    const GraphValueId k = AddLinear(ctx.graph,
                                     normed,
                                     k_proj_weight.shape[0],
                                     k_proj_weight.dtype,
                                     MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionK),
                                     WeightDebugName(TransformerWeightRole::kAttentionK, layer));
    const RawWeightView& v_proj_weight = ctx.weights.layers[layer].attn.v_proj;
    const GraphValueId v = AddLinear(ctx.graph,
                                     normed,
                                     v_proj_weight.shape[0],
                                     v_proj_weight.dtype,
                                     MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionV),
                                     WeightDebugName(TransformerWeightRole::kAttentionV, layer));
    const RoPEOutputs rope = AddRoPE(ctx.graph,
                                     layer,
                                     q,
                                     k,
                                     input.position_ids,
                                     ctx.specs.hidden,
                                     ctx.specs.kv_hidden,
                                     ctx.params.rope,
                                     LayerPrefix(layer) + "self_attn.rotary_emb");
    const KVCachePair cache_out = AddLlamaKVCacheUpdate(ctx, layer, rope.k, v, input.cache);
    const GraphValueId attn = AddAttention(ctx.graph,
                                           layer,
                                           rope.q,
                                           cache_out.k,
                                           cache_out.v,
                                           ctx.specs.hidden,
                                           ctx.params.attention,
                                           LayerPrefix(layer) + "self_attn.attention");
    const RawWeightView& o_proj_weight = ctx.weights.layers[layer].attn.o_proj;
    const GraphValueId o_proj = AddLinear(ctx.graph,
                                          attn,
                                          o_proj_weight.shape[0],
                                          o_proj_weight.dtype,
                                          MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionO),
                                          WeightDebugName(TransformerWeightRole::kAttentionO, layer));
    const GraphValueId residual = AddElementwiseAdd(ctx.graph,
                                                    layer,
                                                    input.hidden,
                                                    o_proj,
                                                    ctx.specs.hidden,
                                                    LayerPrefix(layer) + "post_attention_add");

    AM_CHECK(ctx.graph.GetNodes().size() - block_begin == kAttentionBlockNodeCount,
             "Attention block node count changed unexpectedly");
    return AttentionBlockResult{.hidden = residual};
}

GraphValueId BuildLlamaMlpBlock(LlamaBuildContext& ctx,
                                uint32_t layer,
                                GraphValueId hidden) {
    const size_t block_begin = ctx.graph.GetNodes().size();

    const RawWeightView& post_attn_norm_weight = ctx.weights.layers[layer].norm.post_attn_rmsnorm;
    const GraphValueId normed = AddRmsNorm(ctx.graph,
                                           hidden,
                                           post_attn_norm_weight.dtype,
                                           MakeTransformerWeightBinding(layer, TransformerWeightRole::kPostAttentionNorm),
                                           ctx.params.rms_norm_eps,
                                           WeightDebugName(TransformerWeightRole::kPostAttentionNorm, layer));
    const RawWeightView& gate_weight = ctx.weights.layers[layer].mlp.gate_proj;
    const GraphValueId gate = AddLinear(ctx.graph,
                                        normed,
                                        gate_weight.shape[0],
                                        gate_weight.dtype,
                                        MakeTransformerWeightBinding(layer, TransformerWeightRole::kMlpGate),
                                        WeightDebugName(TransformerWeightRole::kMlpGate, layer));
    const RawWeightView& up_weight = ctx.weights.layers[layer].mlp.up_proj;
    const GraphValueId up = AddLinear(ctx.graph,
                                      normed,
                                      up_weight.shape[0],
                                      up_weight.dtype,
                                      MakeTransformerWeightBinding(layer, TransformerWeightRole::kMlpUp),
                                      WeightDebugName(TransformerWeightRole::kMlpUp, layer));
    const GraphValueId act = AddSiluMul(ctx.graph,
                                        layer,
                                        gate,
                                        up,
                                        ctx.specs.intermediate,
                                        LayerPrefix(layer) + "mlp.act");
    const RawWeightView& down_weight = ctx.weights.layers[layer].mlp.down_proj;
    const GraphValueId down = AddLinear(ctx.graph,
                                        act,
                                        down_weight.shape[0],
                                        down_weight.dtype,
                                        MakeTransformerWeightBinding(layer, TransformerWeightRole::kMlpDown),
                                        WeightDebugName(TransformerWeightRole::kMlpDown, layer));
    const GraphValueId residual = AddElementwiseAdd(ctx.graph,
                                                    layer,
                                                    hidden,
                                                    down,
                                                    ctx.specs.hidden,
                                                    LayerPrefix(layer) + "mlp_add");

    AM_CHECK(ctx.graph.GetNodes().size() - block_begin == kMlpBlockNodeCount,
             "MLP block node count changed unexpectedly");
    return residual;
}

GraphValueId BuildLlamaDecoderLayer(LlamaBuildContext& ctx,
                                    uint32_t layer,
                                    GraphValueId hidden,
                                    GraphValueId position_ids,
                                    KVCachePair cache_in) {
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

    const GraphValueId input_tokens = AddInput(ctx.graph, specs.token_ids, "token_ids");
    const GraphValueId position_ids = AddInput(ctx.graph, specs.position_ids, "position_ids");

    GraphValueId hidden = AddEmbedding(ctx.graph,
                                       input_tokens,
                                       ctx.weights.embed_tokens.shape[0],
                                       ctx.weights.embed_tokens.shape[1],
                                       ctx.weights.embed_tokens.dtype,
                                       MakeTransformerWeightBinding(std::nullopt,
                                                                    TransformerWeightRole::kTokenEmbedding),
                                       WeightDebugName(TransformerWeightRole::kTokenEmbedding, std::nullopt));

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const GraphValueId k_cache = AddState(ctx.graph,
                                              specs.kv_cache,
                                              KVCacheStateBinding{.decoder_layer_index = layer_index,
                                                                  .slot = KVCacheSlot::kKey},
                                              LayerPrefix(layer_index) + "self_attn.k_cache");
        const GraphValueId v_cache = AddState(ctx.graph,
                                              specs.kv_cache,
                                              KVCacheStateBinding{.decoder_layer_index = layer_index,
                                                                  .slot = KVCacheSlot::kValue},
                                              LayerPrefix(layer_index) + "self_attn.v_cache");
        hidden = BuildLlamaDecoderLayer(
                ctx, layer_index, hidden, position_ids,
                KVCachePair{.k = k_cache, .v = v_cache});
    }

    const GraphValueId final_hidden = AddRmsNorm(ctx.graph,
                                                 hidden,
                                                 ctx.weights.final_norm.dtype,
                                                 MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kFinalNorm),
                                                 ctx.params.rms_norm_eps,
                                                 WeightDebugName(TransformerWeightRole::kFinalNorm, std::nullopt));
    const RawWeightView& lm_head_weight = ctx.weights.lm_head.has_value()
                                                  ? *ctx.weights.lm_head
                                                  : ctx.weights.embed_tokens;
    const GraphValueId logits = AddLinear(ctx.graph,
                                          final_hidden,
                                          lm_head_weight.shape[0],
                                          lm_head_weight.dtype,
                                          MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kLmHead),
                                          WeightDebugName(TransformerWeightRole::kLmHead, std::nullopt));
    const GraphValueId output_tokens = aethermind::AddArgmax(ctx.graph,
                                                             std::nullopt,
                                                             logits,
                                                             specs.token_ids,
                                                             -1,
                                                             "argmax");
    ctx.graph.MarkOutput(output_tokens, "output_token_ids");

    AM_RETURN_IF_ERROR(ctx.graph.Validate());
    return graph;
}

}// namespace aethermind
