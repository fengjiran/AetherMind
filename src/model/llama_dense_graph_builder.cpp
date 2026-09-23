#include "aethermind/model/llama_dense_graph_builder.h"

#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/weight/weight_binding_resolver.h"
#include "rope_params_builder.h"
#include "transformer_graph_common.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace aethermind {
namespace {

// Attention block: input_layernorm, q_proj, k_proj, v_proj, rotary_emb,
// kv_cache_update, attention, o_proj, post_attention_add.
constexpr size_t kAttentionBlockNodeCount = 9;

// MLP block: post_attention_layernorm, gate_proj, up_proj, act, down_proj,
// mlp_add.
constexpr size_t kMlpBlockNodeCount = 6;

struct AttentionBlockInput {
    GraphValueId hidden;
    GraphValueId position_ids;
    KVCachePair cache;
};

struct DecoderLayerParams {
    float rms_norm_eps = 0.0F;
    RoPEParams rope;
    AttentionParams attention;
};

StatusOr<GraphValueId> BuildAttentionBlock(ModelGraph& graph,
                                           uint32_t layer,
                                           AttentionBlockInput input,
                                           const DecoderLayerRawWeights& layer_weights,
                                           const DecoderLayerParams& params) {
    const size_t block_begin = graph.GetNodes().size();
    const RawWeightView& q_proj_weight = layer_weights.attn.q_proj;
    const RawWeightView& k_proj_weight = layer_weights.attn.k_proj;
    const RawWeightView& v_proj_weight = layer_weights.attn.v_proj;
    const RawWeightView& o_proj_weight = layer_weights.attn.o_proj;
    const auto norm_dtype = layer_weights.norm.input_rmsnorm.dtype;

    AM_ASSIGN_OR_RETURN(const GraphValueId normed,
                        AddRmsNorm(
                                graph,
                                input.hidden,
                                norm_dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kInputNorm),
                                params.rms_norm_eps,
                                detail::WeightDebugName(TransformerWeightRole::kInputNorm, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId q,
                        AddLinear(
                                graph,
                                normed,
                                q_proj_weight.shape[0],
                                q_proj_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kAttentionQ),
                                detail::WeightDebugName(TransformerWeightRole::kAttentionQ, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId k,
                        AddLinear(
                                graph,
                                normed,
                                k_proj_weight.shape[0],
                                k_proj_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kAttentionK),
                                detail::WeightDebugName(TransformerWeightRole::kAttentionK, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId v,
                        AddLinear(graph,
                                  normed,
                                  v_proj_weight.shape[0],
                                  v_proj_weight.dtype,
                                  MakeTransformerWeightBinding(layer,
                                                               TransformerWeightRole::kAttentionV),
                                  detail::WeightDebugName(TransformerWeightRole::kAttentionV, layer)));
    AM_ASSIGN_OR_RETURN(const RoPEOutputs rope,
                        AddRoPE(graph,
                                layer,
                                q,
                                k,
                                input.position_ids,
                                params.rope,
                                detail::LayerPrefix(layer) + "self_attn.rotary_emb"));
    AM_ASSIGN_OR_RETURN(const KVCachePair cache_out,
                        AddKVCacheUpdate(graph,
                                         layer,
                                         rope.k,
                                         v,
                                         input.cache.k,
                                         input.cache.v,
                                         detail::LayerPrefix(layer) + "self_attn.kv_cache_update"));
    AM_ASSIGN_OR_RETURN(const GraphValueId attn,
                        AddAttention(graph,
                                     layer,
                                     rope.q,
                                     cache_out.k,
                                     cache_out.v,
                                     params.attention,
                                     detail::LayerPrefix(layer) + "self_attn.attention"));
    AM_ASSIGN_OR_RETURN(const GraphValueId o_proj,
                        AddLinear(graph,
                                  attn,
                                  o_proj_weight.shape[0],
                                  o_proj_weight.dtype,
                                  MakeTransformerWeightBinding(layer,
                                                               TransformerWeightRole::kAttentionO),
                                  detail::WeightDebugName(TransformerWeightRole::kAttentionO, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId residual,
                        AddElementwiseAdd(graph,
                                          layer,
                                          input.hidden,
                                          o_proj,
                                          detail::LayerPrefix(layer) + "post_attention_add"));

    AM_CHECK(graph.GetNodes().size() - block_begin == kAttentionBlockNodeCount,
             "Attention block node count changed unexpectedly");
    return residual;
}

StatusOr<GraphValueId> BuildMlpBlock(ModelGraph& graph,
                                     uint32_t layer,
                                     GraphValueId input,
                                     const DecoderLayerRawWeights& layer_weights,
                                     const DecoderLayerParams& params) {
    const size_t block_begin = graph.GetNodes().size();
    const RawWeightView& post_attn_norm_weight = layer_weights.norm.post_attn_rmsnorm;
    const RawWeightView& gate_weight = layer_weights.mlp.gate_proj;
    const RawWeightView& up_weight = layer_weights.mlp.up_proj;
    const RawWeightView& down_weight = layer_weights.mlp.down_proj;

    AM_ASSIGN_OR_RETURN(const GraphValueId normed,
                        AddRmsNorm(
                                graph,
                                input,
                                post_attn_norm_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kPostAttentionNorm),
                                params.rms_norm_eps,
                                detail::WeightDebugName(TransformerWeightRole::kPostAttentionNorm, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId gate,
                        AddLinear(
                                graph,
                                normed,
                                gate_weight.shape[0],
                                gate_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kMlpGate),
                                detail::WeightDebugName(TransformerWeightRole::kMlpGate, layer)));

    AM_ASSIGN_OR_RETURN(const GraphValueId up,
                        AddLinear(
                                graph,
                                normed,
                                up_weight.shape[0],
                                up_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kMlpUp),
                                detail::WeightDebugName(TransformerWeightRole::kMlpUp, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId act,
                        AddSiluMul(
                                graph,
                                layer,
                                gate,
                                up,
                                detail::LayerPrefix(layer) + "mlp.act"));
    AM_ASSIGN_OR_RETURN(const GraphValueId down,
                        AddLinear(
                                graph,
                                act,
                                down_weight.shape[0],
                                down_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kMlpDown),
                                detail::WeightDebugName(TransformerWeightRole::kMlpDown, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId residual,
                        AddElementwiseAdd(
                                graph,
                                layer,
                                input,
                                down,
                                detail::LayerPrefix(layer) + "mlp_add"));

    AM_CHECK(graph.GetNodes().size() - block_begin == kMlpBlockNodeCount,
             "MLP block node count changed unexpectedly");
    return residual;
}

StatusOr<GraphValueId> BuildDecoderLayer(ModelGraph& graph,
                                         const DecoderLayerRawWeights& layer_weights,
                                         const DecoderLayerParams& params,
                                         uint32_t layer,
                                         GraphValueId hidden,
                                         GraphValueId position_ids,
                                         KVCachePair cache_in) {
    AM_ASSIGN_OR_RETURN(const GraphValueId attn,
                        BuildAttentionBlock(graph,
                                            layer,
                                            {.hidden = hidden,
                                             .position_ids = position_ids,
                                             .cache = cache_in},
                                            layer_weights,
                                            params));
    return BuildMlpBlock(graph, layer, attn, layer_weights, params);
}

Status ValidateInputs(const HfModelConfig& config, const ResolvedModelWeights& weights) {
    // Per-family builders remain the authority for RoPE scaling conversion
    // (kNone/kLinear mapping, rejection of unsupported HF variants in
    // MakeRoPEParams). Loading validation accepts rope_scaling by default so
    // that supported/unknown HF algorithms reach MakeRoPEParams
    // rather than being blanket-rejected by the Phase-1 loader policy.
    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(config));
    return HfModelValidator::ValidateResolvedModel(config, weights);
}

} // namespace

StatusOr<ModelGraph> BuildLlamaDense(const HfModelConfig& config,
                                     const ResolvedModelWeights& weights) {
    AM_RETURN_IF_ERROR(ValidateInputs(config, weights));
    const DataType act_dtype = config.weight_dtype_hint.IsUndefined()
                                       ? DataType::Float32()
                                       : config.weight_dtype_hint;
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol kv_len = ShapeSymbol::Create();
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim
                                                  : config.hidden_size / config.num_attention_heads;

    // RoPE semantic conversion must happen before any graph mutation so
    // unsupported HF variants are rejected without producing a partial graph.
    AM_ASSIGN_OR_RETURN(RoPEParams rope_params, detail::MakeRoPEParams(config, head_dim));

    const TensorSpec token_ids_spec{.dtype = DataType::Int(64),
                                    .shape = SymbolicShape({seq_len})};
    const TensorSpec position_ids_spec{.dtype = DataType::Int(64),
                                       .shape = SymbolicShape({seq_len})};
    const TensorSpec kv_cache_spec = detail::KVCacheTensorSpec(
            act_dtype, config.num_key_value_heads, kv_len, head_dim);
    const DecoderLayerParams params{
            .rms_norm_eps = static_cast<float>(config.rms_norm_eps),
            .rope = rope_params,
            .attention = AttentionParams{
                    .num_q_heads = config.num_attention_heads,
                    .num_kv_heads = config.num_key_value_heads,
                    .head_dim = head_dim,
            },
    };

    ModelGraph graph;
    const GraphValueId input_tokens = AddInput(graph, token_ids_spec, "token_ids");
    const GraphValueId position_ids = AddInput(graph, position_ids_spec, "position_ids");

    AM_ASSIGN_OR_RETURN(GraphValueId hidden,
                        AddEmbedding(
                                graph,
                                input_tokens,
                                weights.embed_tokens.shape[0],
                                weights.embed_tokens.shape[1],
                                weights.embed_tokens.dtype,
                                MakeTransformerWeightBinding(std::nullopt,
                                                             TransformerWeightRole::kTokenEmbedding),
                                detail::WeightDebugName(TransformerWeightRole::kTokenEmbedding,
                                                        std::nullopt)));

    for (uint32_t i = 0; i < static_cast<uint32_t>(config.num_hidden_layers); ++i) {
        const GraphValueId k_cache = AddState(graph,
                                              kv_cache_spec,
                                              KVCacheStateBinding{.decoder_layer_index = i,
                                                                  .slot = KVCacheSlot::kKey},
                                              detail::LayerPrefix(i) + "self_attn.k_cache");
        const GraphValueId v_cache = AddState(graph,
                                              kv_cache_spec,
                                              KVCacheStateBinding{.decoder_layer_index = i,
                                                                  .slot = KVCacheSlot::kValue},
                                              detail::LayerPrefix(i) + "self_attn.v_cache");
        const DecoderLayerRawWeights& layer_weights = weights.layers[i];
        AM_ASSIGN_OR_RETURN(hidden, BuildDecoderLayer(
                                            graph, layer_weights, params, i, hidden, position_ids,
                                            KVCachePair{.k = k_cache, .v = v_cache}));
    }

    AM_ASSIGN_OR_RETURN(const GraphValueId final_hidden,
                        AddRmsNorm(
                                graph,
                                hidden,
                                weights.final_norm.dtype,
                                MakeTransformerWeightBinding(std::nullopt,
                                                             TransformerWeightRole::kFinalNorm),
                                params.rms_norm_eps,
                                detail::WeightDebugName(TransformerWeightRole::kFinalNorm,
                                                        std::nullopt)));
    const WeightBinding lm_head_binding = MakeTransformerWeightBinding(
            std::nullopt, TransformerWeightRole::kLmHead);
    const RawWeightView* lm_head_weight = ResolveWeightBinding(lm_head_binding, weights);
    if (lm_head_weight == nullptr) {
        return Status::Internal(
                "BuildLlamaDense: lm_head binding did not resolve");
    }
    AM_ASSIGN_OR_RETURN(const GraphValueId logits,
                        AddLinear(
                                graph,
                                final_hidden,
                                lm_head_weight->shape[0],
                                lm_head_weight->dtype,
                                lm_head_binding,
                                detail::WeightDebugName(TransformerWeightRole::kLmHead, std::nullopt)));
    AM_ASSIGN_OR_RETURN(const GraphValueId output_tokens,
                        AddArgmax(graph,
                                  std::nullopt,
                                  logits,
                                  -1,
                                  "argmax"));
    graph.MarkOutput(output_tokens);
    AM_RETURN_IF_ERROR(graph.Validate());
    return graph;
}

} // namespace aethermind