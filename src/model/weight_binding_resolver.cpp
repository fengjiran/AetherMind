#include "aethermind/model/weight_binding_resolver.h"

#include <optional>

namespace aethermind {
namespace {

/// Layer-scoped roles must carry an in-range decoder_layer_index; graph
/// validation already rejects a missing index (TransformerRoleRequiresLayer),
/// so nullptr here means the binding did not come from a validated graph.
const DecoderLayerRawWeights* LayerAt(const ResolvedModelWeights& resolved,
                                      std::optional<uint32_t> layer) noexcept {
    if (!layer.has_value() || *layer >= resolved.layers.size()) {
        return nullptr;
    }
    return &resolved.layers[*layer];
}

} // namespace

const RawWeightView* ResolveWeightBinding(
        const WeightBinding& binding,
        const ResolvedModelWeights& resolved) noexcept {
    const std::optional<TransformerWeightRole> role =
            TryGetTransformerWeightRole(binding);
    if (!role.has_value()) {
        return nullptr;
    }
    const std::optional<uint32_t> layer = binding.decoder_layer_index;
    switch (*role) {
        case TransformerWeightRole::kTokenEmbedding:
            return &resolved.embed_tokens;
        case TransformerWeightRole::kFinalNorm:
            return &resolved.final_norm;
        case TransformerWeightRole::kLmHead:
            // Tied embeddings reuse embed_tokens when the checkpoint carries no
            // independent lm_head.
            return resolved.lm_head.has_value() ? &*resolved.lm_head
                                                : &resolved.embed_tokens;
        case TransformerWeightRole::kInputNorm:
            if (const auto* l = LayerAt(resolved, layer)) return &l->norm.input_rmsnorm;
            return nullptr;
        case TransformerWeightRole::kPostAttentionNorm:
            if (const auto* l = LayerAt(resolved, layer)) return &l->norm.post_attn_rmsnorm;
            return nullptr;
        case TransformerWeightRole::kAttentionQ:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.q_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionK:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.k_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionV:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.v_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionO:
            if (const auto* l = LayerAt(resolved, layer)) return &l->attn.o_proj;
            return nullptr;
        case TransformerWeightRole::kMlpGate:
            if (const auto* l = LayerAt(resolved, layer)) return &l->mlp.gate_proj;
            return nullptr;
        case TransformerWeightRole::kMlpUp:
            if (const auto* l = LayerAt(resolved, layer)) return &l->mlp.up_proj;
            return nullptr;
        case TransformerWeightRole::kMlpDown:
            if (const auto* l = LayerAt(resolved, layer)) return &l->mlp.down_proj;
            return nullptr;
        case TransformerWeightRole::kMoERouter:
            // Dense checkpoints carry no router weight; MoE is out of the
            // current product scope.
            return nullptr;
    }
    return nullptr;
}

} // namespace aethermind
