#include "aethermind/compiler/packing_request_builder.h"

#include "aethermind/operators/operator_schema.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

const RawWeightView* FindRawWeightByRole(const ResolvedModelWeights& resolved,
                                         std::optional<uint32_t> layer,
                                         TransformerWeightRole role);

StatusOr<std::vector<RawWeightView>> ResolveSingleWeight(
        const ResolvedModelWeights& resolved,
        std::optional<uint32_t> layer,
        TransformerWeightRole role) {
    const RawWeightView* raw = FindRawWeightByRole(resolved, layer, role);
    if (raw == nullptr || !raw->IsValid()) {
        return Status::Internal(
                "BuildWeightPackingRequests: resolved weights are "
                "missing the raw weight for a binding");
    }
    return std::vector<RawWeightView>{*raw};
}

/// @brief Resolves the raw weights referenced by a logical weight binding.
///
/// Direct bindings map to their single Transformer-role weight; composite
/// bindings map to their fixed recipe-ordered components (Q, K, V for QKV;
/// Gate, Up for Gate-Up), which are concatenated during prepacking.
StatusOr<std::vector<RawWeightView>> ResolveWeightComponents(
        const ResolvedModelWeights& resolved,
        const WeightBinding& binding) {
    if (const auto* direct = std::get_if<DirectWeightBinding>(&binding.spec)) {
        if (direct->semantic_role.index() == 0) {
            return Status::Internal(
                    "BuildWeightPackingRequests: weight value has no "
                    "semantic role");
        }
        return ResolveSingleWeight(
                resolved,
                binding.decoder_layer_index,
                std::get<TransformerWeightRole>(direct->semantic_role));
    }

    std::vector<TransformerWeightRole> roles;
    if (std::holds_alternative<QkvWeightBinding>(binding.spec)) {
        roles = {TransformerWeightRole::kAttentionQ,
                 TransformerWeightRole::kAttentionK,
                 TransformerWeightRole::kAttentionV};
    } else if (std::holds_alternative<GateUpWeightBinding>(binding.spec)) {
        roles = {TransformerWeightRole::kMlpGate,
                 TransformerWeightRole::kMlpUp};
    } else {
        return Status::Internal(
                "BuildWeightPackingRequests: unknown weight binding spec");
    }

    if (!binding.decoder_layer_index.has_value()) {
        return Status::Internal(
                "BuildWeightPackingRequests: composite binding has no layer "
                "index");
    }
    std::vector<RawWeightView> components;
    components.reserve(roles.size());
    for (const auto role: roles) {
        auto single = ResolveSingleWeight(
                resolved, binding.decoder_layer_index, role);
        if (!single.ok()) {
            return single.status();
        }
        components.push_back(std::move((*single).front()));
    }
    return components;
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
            // Tied embeddings reuse embed_tokens when the checkpoint carries
            // no independent lm_head, mirroring ModelGraphBuilder.
            return resolved.lm_head.has_value() ? &*resolved.lm_head
                                                : &resolved.embed_tokens;
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

} // namespace

StatusOr<std::vector<WeightPrepackPlanner::Request>> BuildWeightPackingRequests(
        const LoweredGraph& lowered,
        const ResolvedModelWeights& resolved) {
    std::vector<WeightPrepackPlanner::Request> requests;
    // (value_index, selector) pairs already requested; one artifact serves
    // every step consuming the same weight value with the same selector.
    std::unordered_map<uint32_t, std::unordered_set<KernelSelector>> seen_requests;

    for (const LoweredStep& step: lowered.steps()) {
        // Packing requests describe packed weight storage only; steps that
        // consume plain or quantized weights must not enter the planner.
        if (step.spec.selector.weight_format != WeightFormat::kPacked) {
            continue;
        }
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
            if (!seen_requests[value.index].insert(step.spec.selector).second) {
                // A duplicate request would collide on the exact artifact key
                // and fail with AlreadyExists during Store.
                continue;
            }
            auto components = ResolveWeightComponents(resolved, weight->binding);
            if (!components.ok()) {
                return components.status();
            }
            WeightPrepackPlanner::Request request{
                    .op_type = step.spec.op_type,
                    .source_id = lowered.artifact_id(),
                    .value_index = value.index,
                    .binding = weight->binding,
                    .selector = step.spec.selector,
            };
            if (IsCompositeWeightBinding(weight->binding)) {
                request.components = std::move(*components);
            } else {
                request.raw_weight = std::move(components->front());
            }
            requests.push_back(std::move(request));
        }
    }
    return requests;
}

} // namespace aethermind
