#include "aethermind/model/graph/graph_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/graph/graph_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace aethermind {
namespace {

// Attention block: input_layernorm, q_proj, k_proj, v_proj, rotary_emb,
// kv_cache_update, attention, o_proj, post_attention_add.
constexpr size_t kAttentionBlockNodeCount = 9;

// MLP block: post_attention_layernorm, gate_proj, up_proj, act, down_proj,
// mlp_add.
constexpr size_t kMlpBlockNodeCount = 6;

TensorSpec KVCacheTensorSpec(DataType dtype, int64_t num_kv_heads,
                             ShapeSymbol cache_len, int64_t head_dim) {
    return TensorSpec{.dtype = dtype,
                      .shape = {ShapeSymbol::CreateFromValue(num_kv_heads),
                                cache_len,
                                ShapeSymbol::CreateFromValue(head_dim)}};
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

struct AttentionBlockInput {
    GraphValueId hidden;
    GraphValueId position_ids;
    KVCachePair cache;
};

struct LlamaBuildParams {
    float rms_norm_eps = 0.0F;
    RoPEParams rope;
    AttentionParams attention;
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

StatusOr<GraphValueId> BuildAttentionBlock(ModelGraph& graph,
                                           uint32_t layer,
                                           AttentionBlockInput input,
                                           const DecoderLayerRawWeights& layer_weights,
                                           const LlamaBuildParams& params) {
    const size_t block_begin = graph.GetNodes().size();
    const RawWeightView& q_proj_weight = layer_weights.attn.q_proj;
    const RawWeightView& k_proj_weight = layer_weights.attn.k_proj;
    const RawWeightView& v_proj_weight = layer_weights.attn.v_proj;
    const RawWeightView& o_proj_weight = layer_weights.attn.o_proj;
    const auto norm_dtype = layer_weights.norm.input_rmsnorm.dtype;

    AM_ASSIGN_OR_RETURN(const GraphValueId normed, AddRmsNorm(
                                                           graph,
                                                           input.hidden,
                                                           norm_dtype,
                                                           MakeTransformerWeightBinding(layer, TransformerWeightRole::kInputNorm),
                                                           params.rms_norm_eps,
                                                           WeightDebugName(TransformerWeightRole::kInputNorm, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId q, AddLinear(
                                                      graph,
                                                      normed,
                                                      q_proj_weight.shape[0],
                                                      q_proj_weight.dtype,
                                                      MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionQ),
                                                      WeightDebugName(TransformerWeightRole::kAttentionQ, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId k, AddLinear(
                                                      graph,
                                                      normed,
                                                      k_proj_weight.shape[0],
                                                      k_proj_weight.dtype,
                                                      MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionK),
                                                      WeightDebugName(TransformerWeightRole::kAttentionK, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId v, AddLinear(
                                                      graph,
                                                      normed,
                                                      v_proj_weight.shape[0],
                                                      v_proj_weight.dtype,
                                                      MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionV),
                                                      WeightDebugName(TransformerWeightRole::kAttentionV, layer)));
    AM_ASSIGN_OR_RETURN(const RoPEOutputs rope, AddRoPE(
                                                        graph,
                                                        layer,
                                                        q,
                                                        k,
                                                        input.position_ids,
                                                        params.rope,
                                                        LayerPrefix(layer) + "self_attn.rotary_emb"));
    AM_ASSIGN_OR_RETURN(const KVCachePair cache_out, AddKVCacheUpdate(
                                                             graph,
                                                             layer,
                                                             rope.k,
                                                             v,
                                                             input.cache.k,
                                                             input.cache.v,
                                                             LayerPrefix(layer) + "self_attn.kv_cache_update"));
    AM_ASSIGN_OR_RETURN(const GraphValueId attn, AddAttention(
                                                         graph,
                                                         layer,
                                                         rope.q,
                                                         cache_out.k,
                                                         cache_out.v,
                                                         params.attention,
                                                         LayerPrefix(layer) + "self_attn.attention"));
    AM_ASSIGN_OR_RETURN(const GraphValueId o_proj, AddLinear(
                                                           graph,
                                                           attn,
                                                           o_proj_weight.shape[0],
                                                           o_proj_weight.dtype,
                                                           MakeTransformerWeightBinding(layer, TransformerWeightRole::kAttentionO),
                                                           WeightDebugName(TransformerWeightRole::kAttentionO, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId residual, AddElementwiseAdd(
                                                             graph,
                                                             layer,
                                                             input.hidden,
                                                             o_proj,
                                                             LayerPrefix(layer) + "post_attention_add"));

    AM_CHECK(graph.GetNodes().size() - block_begin == kAttentionBlockNodeCount,
             "Attention block node count changed unexpectedly");
    return residual;
}

StatusOr<GraphValueId> BuildMlpBlock(ModelGraph& graph,
                                     uint32_t layer,
                                     GraphValueId input,
                                     const DecoderLayerRawWeights& layer_weights,
                                     const LlamaBuildParams& params) {
    const size_t block_begin = graph.GetNodes().size();
    const RawWeightView& post_attn_norm_weight = layer_weights.norm.post_attn_rmsnorm;
    const RawWeightView& gate_weight = layer_weights.mlp.gate_proj;
    const RawWeightView& up_weight = layer_weights.mlp.up_proj;
    const RawWeightView& down_weight = layer_weights.mlp.down_proj;

    AM_ASSIGN_OR_RETURN(const GraphValueId normed, AddRmsNorm(
                                                           graph,
                                                           input,
                                                           post_attn_norm_weight.dtype,
                                                           MakeTransformerWeightBinding(layer, TransformerWeightRole::kPostAttentionNorm),
                                                           params.rms_norm_eps,
                                                           WeightDebugName(TransformerWeightRole::kPostAttentionNorm, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId gate, AddLinear(
                                                         graph,
                                                         normed,
                                                         gate_weight.shape[0],
                                                         gate_weight.dtype,
                                                         MakeTransformerWeightBinding(layer, TransformerWeightRole::kMlpGate),
                                                         WeightDebugName(TransformerWeightRole::kMlpGate, layer)));

    AM_ASSIGN_OR_RETURN(const GraphValueId up, AddLinear(
                                                       graph,
                                                       normed,
                                                       up_weight.shape[0],
                                                       up_weight.dtype,
                                                       MakeTransformerWeightBinding(layer, TransformerWeightRole::kMlpUp),
                                                       WeightDebugName(TransformerWeightRole::kMlpUp, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId act, AddSiluMul(
                                                        graph,
                                                        layer,
                                                        gate,
                                                        up,
                                                        LayerPrefix(layer) + "mlp.act"));
    AM_ASSIGN_OR_RETURN(const GraphValueId down, AddLinear(
                                                         graph,
                                                         act,
                                                         down_weight.shape[0],
                                                         down_weight.dtype,
                                                         MakeTransformerWeightBinding(layer, TransformerWeightRole::kMlpDown),
                                                         WeightDebugName(TransformerWeightRole::kMlpDown, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId residual, AddElementwiseAdd(
                                                             graph,
                                                             layer,
                                                             input,
                                                             down,
                                                             LayerPrefix(layer) + "mlp_add"));

    AM_CHECK(graph.GetNodes().size() - block_begin == kMlpBlockNodeCount,
             "MLP block node count changed unexpectedly");
    return residual;
}

StatusOr<GraphValueId> BuildDecoderLayer(ModelGraph& graph,
                                         const DecoderLayerRawWeights& layer_weights,
                                         const LlamaBuildParams& params,
                                         uint32_t layer,
                                         GraphValueId hidden,
                                         GraphValueId position_ids,
                                         KVCachePair cache_in) {
    AM_ASSIGN_OR_RETURN(const GraphValueId attn, BuildAttentionBlock(
                                                         graph,
                                                         layer,
                                                         AttentionBlockInput{.hidden = hidden,
                                                                             .position_ids = position_ids,
                                                                             .cache = cache_in},
                                                         layer_weights,
                                                         params));
    return BuildMlpBlock(graph, layer, attn, layer_weights, params);
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
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim
                                                  : config.hidden_size / config.num_attention_heads;

    const TensorSpec token_ids_spec{.dtype = DataType::Int(64),
                                    .shape = SymbolicShape({seq_len})};
    const TensorSpec position_ids_spec{.dtype = DataType::Int(64),
                                       .shape = SymbolicShape({seq_len})};
    const TensorSpec kv_cache_spec = KVCacheTensorSpec(
            act_dtype, config.num_key_value_heads, kv_len, head_dim);
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
    const GraphValueId input_tokens = AddInput(graph, token_ids_spec, "token_ids");
    const GraphValueId position_ids = AddInput(graph, position_ids_spec, "position_ids");

    AM_ASSIGN_OR_RETURN(GraphValueId hidden, AddEmbedding(
                                                     graph,
                                                     input_tokens,
                                                     weights.embed_tokens.shape[0],
                                                     weights.embed_tokens.shape[1],
                                                     weights.embed_tokens.dtype,
                                                     MakeTransformerWeightBinding(std::nullopt,
                                                                                  TransformerWeightRole::kTokenEmbedding),
                                                     WeightDebugName(TransformerWeightRole::kTokenEmbedding, std::nullopt)));

    for (uint32_t layer_index = 0; layer_index < static_cast<uint32_t>(config.num_hidden_layers); ++layer_index) {
        const GraphValueId k_cache = AddState(graph,
                                              kv_cache_spec,
                                              KVCacheStateBinding{.decoder_layer_index = layer_index,
                                                                  .slot = KVCacheSlot::kKey},
                                              LayerPrefix(layer_index) + "self_attn.k_cache");
        const GraphValueId v_cache = AddState(graph,
                                              kv_cache_spec,
                                              KVCacheStateBinding{.decoder_layer_index = layer_index,
                                                                  .slot = KVCacheSlot::kValue},
                                              LayerPrefix(layer_index) + "self_attn.v_cache");
        const DecoderLayerRawWeights& layer_weights = weights.layers[layer_index];
        AM_ASSIGN_OR_RETURN(hidden, BuildDecoderLayer(
                                            graph, layer_weights, params, layer_index, hidden, position_ids,
                                            KVCachePair{.k = k_cache, .v = v_cache}));
    }

    AM_ASSIGN_OR_RETURN(const GraphValueId final_hidden, AddRmsNorm(
                                                                 graph,
                                                                 hidden,
                                                                 weights.final_norm.dtype,
                                                                 MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kFinalNorm),
                                                                 params.rms_norm_eps,
                                                                 WeightDebugName(TransformerWeightRole::kFinalNorm, std::nullopt)));
    const RawWeightView& lm_head_weight = weights.lm_head.has_value()
                                                  ? *weights.lm_head
                                                  : weights.embed_tokens;
    AM_ASSIGN_OR_RETURN(const GraphValueId logits, AddLinear(
                                                           graph,
                                                           final_hidden,
                                                           lm_head_weight.shape[0],
                                                           lm_head_weight.dtype,
                                                           MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kLmHead),
                                                           WeightDebugName(TransformerWeightRole::kLmHead, std::nullopt)));
    AM_ASSIGN_OR_RETURN(const GraphValueId output_tokens, AddArgmax(graph,
                                                                    std::nullopt,
                                                                    logits,
                                                                    -1,
                                                                    "argmax"));
    graph.MarkOutput(output_tokens);

    AM_RETURN_IF_ERROR(graph.Validate());
    return graph;
}

}// namespace aethermind
