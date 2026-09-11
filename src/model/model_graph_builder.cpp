#include "aethermind/model/model_graph_builder.h"
#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/operators/rope_frequency_resolver.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
    return {.dtype = dtype,
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

static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kStandard) == static_cast<uint8_t>(RoPEAlgorithm::kStandard));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kLinear) == static_cast<uint8_t>(RoPEAlgorithm::kLinear));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kDynamicNtk) == static_cast<uint8_t>(RoPEAlgorithm::kDynamicNtk));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kYarn) == static_cast<uint8_t>(RoPEAlgorithm::kYarn));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kLlama3) == static_cast<uint8_t>(RoPEAlgorithm::kLlama3));
static_assert(static_cast<uint8_t>(HfRoPEAlgorithm::kLongRope) == static_cast<uint8_t>(RoPEAlgorithm::kLongRope));

StatusOr<double> RequireRopeDouble(const std::optional<double>& value,
                                   std::string_view field) {
    if (!value.has_value() || !std::isfinite(*value)) {
        return Status::InvalidArgument("ModelGraphBuilder: RoPE '" + std::string(field) +
                                       "' must be provided and finite");
    }
    return *value;
}

StatusOr<int64_t> RequireRopeContext(const std::optional<int64_t>& value) {
    if (!value.has_value() || *value <= 0) {
        return Status::InvalidArgument(
                "ModelGraphBuilder: RoPE 'original_max_position_embeddings' "
                "must be provided and positive");
    }
    return *value;
}

StatusOr<int64_t> ResolveHfRotaryDim(const HfRopeConfig& rope, int64_t head_dim) {
    if (!rope.partial_rotary_factor.has_value()) {
        return rope.rotary_dim.value_or(head_dim);
    }

    const double factor = *rope.partial_rotary_factor;
    const double dimension = factor * static_cast<double>(head_dim);
    if (!std::isfinite(factor) || factor <= 0.0 || factor > 1.0 ||
        !std::isfinite(dimension) || dimension < 1.0 ||
        dimension > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return Status::InvalidArgument("ModelGraphBuilder: invalid partial_rotary_factor");
    }

    const auto derived = static_cast<int64_t>(dimension);
    if (rope.rotary_dim.has_value() && *rope.rotary_dim != derived) {
        return Status::InvalidArgument(
                "ModelGraphBuilder: rotary_dim conflicts with partial_rotary_factor");
    }
    return derived;
}

double Mscale(double factor, double multiplier) noexcept {
    return factor <= 1.0 || multiplier == 0.0 ? 1.0 : 0.1 * multiplier * std::log(factor) + 1.0;
}

// This is the single HF-to-semantic conversion boundary. Optional HF fields
// and aliases are normalized into one complete algorithm variant here.
StatusOr<RoPEParams> MakeRoPEParams(const HfModelConfig& config, int64_t head_dim) {
    AM_ASSIGN_OR_RETURN(const int64_t rotary_dim, ResolveHfRotaryDim(config.rope, head_dim));
    RoPEAlgorithmParams algorithm = StandardRoPE{};
    switch (config.rope.algorithm) {
        case HfRoPEAlgorithm::kStandard:
            break;
        case HfRoPEAlgorithm::kLinear: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            algorithm = LinearRoPE{.factor = factor};
            break;
        }
        case HfRoPEAlgorithm::kDynamicNtk: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            algorithm = DynamicNtkRoPE{
                    .factor = factor,
                    .original_context_length = config.rope.original_context_length.value_or(
                            config.max_position_embeddings)};
            break;
        }
        case HfRoPEAlgorithm::kYarn: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            AM_ASSIGN_OR_RETURN(const int64_t original_context,
                                RequireRopeContext(config.rope.original_context_length));
            double rotary_output_scale = Mscale(factor, 1.0);
            if (config.rope.attention_factor.has_value()) {
                rotary_output_scale = *config.rope.attention_factor;
            } else if (config.rope.mscale.has_value() && config.rope.mscale_all_dim.has_value() &&
                       *config.rope.mscale != 0.0 && *config.rope.mscale_all_dim != 0.0) {
                rotary_output_scale = Mscale(factor, *config.rope.mscale) /
                                      Mscale(factor, *config.rope.mscale_all_dim);
            }
            algorithm = YarnRoPE{.factor = factor,
                                 .original_context_length = original_context,
                                 .beta_fast = config.rope.beta_fast.value_or(32.0),
                                 .beta_slow = config.rope.beta_slow.value_or(1.0),
                                 .rotary_output_scale = rotary_output_scale,
                                 .truncate_correction_range =
                                         config.rope.truncate_correction_range.value_or(true)};
            break;
        }
        case HfRoPEAlgorithm::kLlama3: {
            AM_ASSIGN_OR_RETURN(const double factor,
                                RequireRopeDouble(config.rope.factor, "factor"));
            AM_ASSIGN_OR_RETURN(const int64_t original_context,
                                RequireRopeContext(config.rope.original_context_length));
            AM_ASSIGN_OR_RETURN(const double low,
                                RequireRopeDouble(config.rope.low_frequency_factor,
                                                  "low_freq_factor"));
            AM_ASSIGN_OR_RETURN(const double high,
                                RequireRopeDouble(config.rope.high_frequency_factor,
                                                  "high_freq_factor"));
            algorithm = Llama3RoPE{.factor = factor,
                                   .low_frequency_factor = low,
                                   .high_frequency_factor = high,
                                   .original_context_length = original_context};
            break;
        }
        case HfRoPEAlgorithm::kLongRope:
        case HfRoPEAlgorithm::kSu: {
            AM_ASSIGN_OR_RETURN(const int64_t original_context,
                                RequireRopeContext(config.rope.original_context_length));
            if (config.rope.short_factors.empty() || config.rope.long_factors.empty()) {
                return Status::InvalidArgument("ModelGraphBuilder: LongRoPE "
                                               "requires short_factor and long_factor arrays");
            }

            double rotary_output_scale = 1.0;
            if (config.rope.attention_factor.has_value()) {
                rotary_output_scale = *config.rope.attention_factor;
            } else if (config.rope.factor.has_value()) {
                const double factor = *config.rope.factor;
                rotary_output_scale = std::sqrt(
                        1.0 + std::log(factor) / std::log(static_cast<double>(original_context)));
            }
            algorithm = LongRoPE{.short_factors = config.rope.short_factors,
                                 .long_factors = config.rope.long_factors,
                                 .original_context_length = original_context,
                                 .rotary_output_scale = rotary_output_scale};
            break;
        }
        case HfRoPEAlgorithm::kUnknown:
            return Status::InvalidArgument(
                    "ModelGraphBuilder::BuildLlamaDense: HF RoPE algorithm '" +
                    std::string(ToString(config.rope.algorithm)) + "' is not supported");
    }
    RoPEParams result{
            .head_dim = head_dim,
            .rotary_dim = rotary_dim,
            .num_q_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .max_pos_embeddings = config.max_position_embeddings,
            .theta = config.rope.theta,
            .algorithm = std::move(algorithm),
    };
    AM_RETURN_IF_ERROR(ValidateRoPEFreqParams(result));
    return result;
}

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
                                WeightDebugName(TransformerWeightRole::kInputNorm, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId q,
                        AddLinear(
                                graph,
                                normed,
                                q_proj_weight.shape[0],
                                q_proj_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kAttentionQ),
                                WeightDebugName(TransformerWeightRole::kAttentionQ, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId k,
                        AddLinear(
                                graph,
                                normed,
                                k_proj_weight.shape[0],
                                k_proj_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kAttentionK),
                                WeightDebugName(TransformerWeightRole::kAttentionK, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId v,
                        AddLinear(graph,
                                  normed,
                                  v_proj_weight.shape[0],
                                  v_proj_weight.dtype,
                                  MakeTransformerWeightBinding(layer,
                                                               TransformerWeightRole::kAttentionV),
                                  WeightDebugName(TransformerWeightRole::kAttentionV, layer)));
    AM_ASSIGN_OR_RETURN(const RoPEOutputs rope,
                        AddRoPE(graph,
                                layer,
                                q,
                                k,
                                input.position_ids,
                                params.rope,
                                LayerPrefix(layer) + "self_attn.rotary_emb"));
    AM_ASSIGN_OR_RETURN(const KVCachePair cache_out,
                        AddKVCacheUpdate(graph,
                                         layer,
                                         rope.k,
                                         v,
                                         input.cache.k,
                                         input.cache.v,
                                         LayerPrefix(layer) + "self_attn.kv_cache_update"));
    AM_ASSIGN_OR_RETURN(const GraphValueId attn,
                        AddAttention(graph,
                                     layer,
                                     rope.q,
                                     cache_out.k,
                                     cache_out.v,
                                     params.attention,
                                     LayerPrefix(layer) + "self_attn.attention"));
    AM_ASSIGN_OR_RETURN(const GraphValueId o_proj,
                        AddLinear(graph,
                                  attn,
                                  o_proj_weight.shape[0],
                                  o_proj_weight.dtype,
                                  MakeTransformerWeightBinding(layer,
                                                               TransformerWeightRole::kAttentionO),
                                  WeightDebugName(TransformerWeightRole::kAttentionO, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId residual,
                        AddElementwiseAdd(graph,
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
                                WeightDebugName(TransformerWeightRole::kPostAttentionNorm, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId gate,
                        AddLinear(
                                graph,
                                normed,
                                gate_weight.shape[0],
                                gate_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kMlpGate),
                                WeightDebugName(TransformerWeightRole::kMlpGate, layer)));

    AM_ASSIGN_OR_RETURN(const GraphValueId up,
                        AddLinear(
                                graph,
                                normed,
                                up_weight.shape[0],
                                up_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kMlpUp),
                                WeightDebugName(TransformerWeightRole::kMlpUp, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId act,
                        AddSiluMul(
                                graph,
                                layer,
                                gate,
                                up,
                                LayerPrefix(layer) + "mlp.act"));
    AM_ASSIGN_OR_RETURN(const GraphValueId down,
                        AddLinear(
                                graph,
                                act,
                                down_weight.shape[0],
                                down_weight.dtype,
                                MakeTransformerWeightBinding(layer,
                                                             TransformerWeightRole::kMlpDown),
                                WeightDebugName(TransformerWeightRole::kMlpDown, layer)));
    AM_ASSIGN_OR_RETURN(const GraphValueId residual,
                        AddElementwiseAdd(
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
    // The builder is the single authority for RoPE scaling conversion
    // (kNone/kLinear mapping, rejection of unsupported HF variants in
    // MakeRoPEParams). Loading validation accepts rope_scaling by default so
    // that supported/unknown HF algorithms reach MakeRoPEParams
    // rather than being blanket-rejected by the Phase-1 loader policy.
    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(config));
    return HfModelValidator::ValidateResolvedModel(config, weights);
}

} // namespace

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

    // RoPE semantic conversion must happen before any graph mutation so
    // unsupported HF variants are rejected without producing a partial graph.
    AM_ASSIGN_OR_RETURN(RoPEParams rope_params, MakeRoPEParams(config, head_dim));

    const TensorSpec token_ids_spec{.dtype = DataType::Int(64),
                                    .shape = SymbolicShape({seq_len})};
    const TensorSpec position_ids_spec{.dtype = DataType::Int(64),
                                       .shape = SymbolicShape({seq_len})};
    const TensorSpec kv_cache_spec = KVCacheTensorSpec(
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
                                WeightDebugName(TransformerWeightRole::kTokenEmbedding,
                                                std::nullopt)));

    for (uint32_t i = 0; i < static_cast<uint32_t>(config.num_hidden_layers); ++i) {
        const GraphValueId k_cache = AddState(graph,
                                              kv_cache_spec,
                                              KVCacheStateBinding{.decoder_layer_index = i,
                                                                  .slot = KVCacheSlot::kKey},
                                              LayerPrefix(i) + "self_attn.k_cache");
        const GraphValueId v_cache = AddState(graph,
                                              kv_cache_spec,
                                              KVCacheStateBinding{.decoder_layer_index = i,
                                                                  .slot = KVCacheSlot::kValue},
                                              LayerPrefix(i) + "self_attn.v_cache");
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
                                WeightDebugName(TransformerWeightRole::kFinalNorm,
                                                std::nullopt)));
    const RawWeightView& lm_head_weight = weights.lm_head.has_value()
                                                  ? *weights.lm_head
                                                  : weights.embed_tokens;
    AM_ASSIGN_OR_RETURN(const GraphValueId logits,
                        AddLinear(
                                graph,
                                final_hidden,
                                lm_head_weight.shape[0],
                                lm_head_weight.dtype,
                                MakeTransformerWeightBinding(std::nullopt,
                                                             TransformerWeightRole::kLmHead),
                                WeightDebugName(TransformerWeightRole::kLmHead, std::nullopt)));
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
