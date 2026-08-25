#include "aethermind/compiler/packing_request_builder.h"

#include "aethermind/operators/operator_schema.h"

#include <string>

namespace aethermind {
namespace {

StatusOr<TransformerWeightRole> ResolveRole(const WeightBinding& binding) {
    const auto* direct = std::get_if<DirectWeightBinding>(&binding.spec);
    if (direct == nullptr) {
        return Status::Internal(
                "BuildWeightPackingRequests: composite weight bindings are not "
                "supported yet");
    }
    if (direct->semantic_role.index() == 0) {
        return Status::Internal(
                "BuildWeightPackingRequests: weight value has no semantic role");
    }
    return std::get<TransformerWeightRole>(direct->semantic_role);
}

const RawWeightView* FindRawWeightByRole(const ResolvedModelWeights& resolved,
                                         std::optional<uint32_t> layer,
                                         TransformerWeightRole role) {
    const auto layer_at = [&](size_t index) -> const DecoderLayerRawWeights* {
        if (index >= resolved.layers.size()) {
            return nullptr;
        }
        return &resolved.layers[index];
    };
    switch (role) {
        case TransformerWeightRole::kTokenEmbedding:
            return &resolved.embed_tokens;
        case TransformerWeightRole::kFinalNorm:
            return &resolved.final_norm;
        case TransformerWeightRole::kLmHead:
            return resolved.lm_head.has_value() ? &*resolved.lm_head : nullptr;
        case TransformerWeightRole::kInputNorm:
            return layer.has_value() && *layer < resolved.layers.size()
                           ? &resolved.layers[*layer].norm.input_rmsnorm
                           : nullptr;
        case TransformerWeightRole::kPostAttentionNorm:
            return layer.has_value() && *layer < resolved.layers.size()
                           ? &resolved.layers[*layer].norm.post_attn_rmsnorm
                           : nullptr;
        case TransformerWeightRole::kAttentionQ:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->attn.q_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionK:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->attn.k_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionV:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->attn.v_proj;
            return nullptr;
        case TransformerWeightRole::kAttentionO:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->attn.o_proj;
            return nullptr;
        case TransformerWeightRole::kMlpGate:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->mlp.gate_proj;
            return nullptr;
        case TransformerWeightRole::kMlpUp:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->mlp.up_proj;
            return nullptr;
        case TransformerWeightRole::kMlpDown:
            if (const auto* l = layer_at(layer.value_or(0))) return &l->mlp.down_proj;
            return nullptr;
        case TransformerWeightRole::kMoERouter:
            return nullptr;
    }
    return nullptr;
}

}// namespace

StatusOr<std::vector<WeightPrepackPlanner::Request>> BuildWeightPackingRequests(
        const LoweredGraph& lowered,
        const ResolvedModelWeights& resolved) {
    std::vector<WeightPrepackPlanner::Request> requests;

    for (const LoweredStep& step: lowered.steps()) {
        const auto schema = GetOperatorSchema(step.spec.op_type);
        if (!schema.ok()) {
            return schema.status();
        }
        for (size_t port = 0; port < schema->input_ports.size(); ++port) {
            if (schema->input_ports[port].kind != OperatorPortKind::kWeight) {
                continue;
            }
            if (port >= step.binding.input_values.size()) {
                return Status::Internal(
                        "BuildWeightPackingRequests: kWeight port beyond binding");
            }
            const GraphValueId value = step.binding.input_values[port];
            if (value.index >= lowered.values().size()) {
                return Status::Internal(
                        "BuildWeightPackingRequests: weight value out of range");
            }
            const auto* weight =
                    std::get_if<WeightValue>(&lowered.values()[value.index].payload);
            if (weight == nullptr) {
                return Status::Internal(
                        "BuildWeightPackingRequests: kWeight value has no "
                        "WeightValue payload");
            }
            auto role = ResolveRole(weight->binding);
            if (!role.ok()) {
                return role.status();
            }
            const RawWeightView* raw = FindRawWeightByRole(
                    resolved, weight->binding.decoder_layer_index, *role);
            if (raw == nullptr || !raw->IsValid()) {
                return Status::Internal(
                        "BuildWeightPackingRequests: resolved weights are "
                        "missing the raw weight for a binding");
            }
            requests.push_back(WeightPrepackPlanner::Request{
                    .op_type = step.spec.op_type,
                    .source_id = lowered.artifact_id(),
                    .value_index = value.index,
                    .binding = weight->binding,
                    .raw_weight = *raw,
                    .selector = step.spec.selector,
            });
        }
    }
    return requests;
}

}// namespace aethermind
