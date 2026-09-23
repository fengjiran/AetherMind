#include "transformer_graph_common.h"

namespace aethermind::detail {

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

TensorSpec KVCacheTensorSpec(DataType dtype, int64_t num_kv_heads,
                             ShapeSymbol cache_len, int64_t head_dim) {
    return {.dtype = dtype,
            .shape = {ShapeSymbol::CreateFromValue(num_kv_heads),
                      cache_len,
                      ShapeSymbol::CreateFromValue(head_dim)}};
}

} // namespace aethermind::detail