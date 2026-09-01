#include "aethermind/graph/optimization/qkv_linear_fusion_pass.h"
#include "aethermind/operators/operator_inference.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

constexpr size_t kQueryProjection = 0U;
constexpr size_t kKeyProjection = 1U;
constexpr size_t kValueProjection = 2U;
constexpr size_t kProjectionCount = 3U;

/// @brief A single Linear node which is eligible to participate in QKV fusion.
struct LinearProjectionCandidate {
    GraphNodeId node{};
    GraphValueId input{};
    GraphValueId output{};
    uint32_t decoder_layer_index = 0;
    TransformerWeightRole role = TransformerWeightRole::kAttentionQ;
    TensorSpec input_spec{};
    TensorSpec weight_spec{};
    GraphValueDesc output_desc{};
    QuantizationSpec weight_quantization{};
    int64_t output_features = 0;
};

/// @brief A topology-ordered QKV bucket. It may be incomplete or ambiguous;
/// only QkvLinearPattern is allowed to reach the rewriting routine.
struct QkvCandidateGroup {
    GraphValueId input{};
    std::array<std::optional<LinearProjectionCandidate>, kProjectionCount> projections{};
    uint32_t decoder_layer_index = 0;
    bool ambiguous = false;
};

/// @brief The projection data needed by the mechanical rewrite step.
struct ProjectionMatch {
    GraphNodeId node{};
    GraphValueId output{};
    GraphValueDesc output_desc{};
    int64_t output_features = 0;
};

/// @brief A fully validated, directly rewritable QKV projection pattern.
struct QkvLinearPattern {
    ProjectionMatch query{};
    ProjectionMatch key{};
    ProjectionMatch value{};
    GraphValueId input{};
    uint32_t decoder_layer_index = 0;
    TensorSpec fused_weight_spec{};
    QuantizationSpec weight_quantization{};
};

std::optional<size_t> ProjectionIndex(TransformerWeightRole role) noexcept {
    switch (role) {
        case TransformerWeightRole::kAttentionQ:
            return kQueryProjection;
        case TransformerWeightRole::kAttentionK:
            return kKeyProjection;
        case TransformerWeightRole::kAttentionV:
            return kValueProjection;
        default:
            return std::nullopt;
    }
}

bool IsRankTwo(const TensorSpec& spec) noexcept {
    return spec.shape.IsRanked() && spec.shape.rank() == 2U;
}

bool HasConsistentLinearMetadata(const GraphNodeView& node,
                                 const GraphValueDesc& input,
                                 const GraphValueDesc& weight,
                                 const GraphValueDesc& output) {
    if (!std::holds_alternative<ActivationValue>(output.payload)) {
        return false;
    }

    const std::array<TensorSpec, 2> inputs{input.spec, weight.spec};
    StatusOr<InferenceResult> inferred = InferOperator(
            OpType::kLinear, node.op_params, std::span<const TensorSpec>{inputs});
    return inferred.ok() && inferred->outputs.size() == 1U &&
           inferred->outputs[0] == output.spec;
}

StatusOr<std::optional<LinearProjectionCandidate>> FindLinearProjectionCandidate(
        GraphRewriteSession& session,
        GraphNodeId node_id) {
    if (!session.IsNodeLive(node_id)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    StatusOr<GraphNodeView> node = session.GetNodeView(node_id);
    AM_RETURN_IF_ERROR(node.status());
    if (!(node->op_type == OpType::kLinear &&
          node->inputs.size() == 2U &&
          node->outputs.size() == 1U &&
          std::holds_alternative<LinearParams>(node->op_params) &&
          node->attrs.bytes.empty() &&
          node->decoder_layer_index.has_value())) {
        return std::optional<LinearProjectionCandidate>{};
    }

    const GraphValueId input = node->inputs[0];
    const GraphValueId weight = node->inputs[1];
    const GraphValueId output = node->outputs[0];
    if (session.GetResolvedValue(output) != output || !session.IsValueLive(output)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    StatusOr<bool> has_live_output = session.HasLiveConsumers(output);
    AM_RETURN_IF_ERROR(has_live_output.status());
    if (!session.IsGraphOutput(output) && !*has_live_output) {
        return std::optional<LinearProjectionCandidate>{};
    }

    StatusOr<GraphValueDesc> input_desc = session.GetValueOutputMetadata(input);
    AM_RETURN_IF_ERROR(input_desc.status());
    StatusOr<GraphValueDesc> weight_desc = session.GetValueOutputMetadata(weight);
    AM_RETURN_IF_ERROR(weight_desc.status());
    StatusOr<GraphValueDesc> output_desc = session.GetValueOutputMetadata(output);
    AM_RETURN_IF_ERROR(output_desc.status());

    if (!std::holds_alternative<ActivationValue>(input_desc->payload)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    const auto* weight_value = std::get_if<WeightValue>(&weight_desc->payload);
    if (weight_value == nullptr) {
        return std::optional<LinearProjectionCandidate>{};
    }

    const auto* direct = TryGetDirectWeightBinding(weight_value->binding);
    const auto role = TryGetTransformerWeightRole(weight_value->binding);
    if (direct == nullptr || !role.has_value() ||
        !ProjectionIndex(*role).has_value() ||
        direct->slot != ParameterSlot::kKernel ||
        weight_value->binding.decoder_layer_index != node->decoder_layer_index ||
        !IsRankTwo(weight_desc->spec)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    const ShapeSymbol& output_features = weight_desc->spec.shape[0];
    if (!output_features.IsStatic() || output_features.GetStaticValue() <= 0 ||
        !HasConsistentLinearMetadata(*node, *input_desc, *weight_desc, *output_desc)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    return std::optional{LinearProjectionCandidate{
            .node = node_id,
            .input = input,
            .output = output,
            .decoder_layer_index = *node->decoder_layer_index,
            .role = *role,
            .input_spec = input_desc->spec,
            .weight_spec = weight_desc->spec,
            .output_desc = std::move(*output_desc),
            .weight_quantization = weight_desc->quantization,
            .output_features = output_features.GetStaticValue(),
    }};
}

void AddToCandidateGroups(std::vector<QkvCandidateGroup>& groups,
                          LinearProjectionCandidate candidate) {
    QkvCandidateGroup* group = nullptr;
    for (auto& existing: groups) {
        if (existing.input == candidate.input &&
            existing.decoder_layer_index == candidate.decoder_layer_index) {
            group = &existing;
            break;
        }
    }

    if (group == nullptr) {
        groups.push_back(QkvCandidateGroup{
                .input = candidate.input,
                .decoder_layer_index = candidate.decoder_layer_index,
        });
        group = &groups.back();
    }

    const size_t projection_index = *ProjectionIndex(candidate.role);
    if (group->projections[projection_index].has_value()) {
        group->ambiguous = true;
        return;
    }
    group->projections[projection_index] = std::move(candidate);
}

std::optional<QkvLinearPattern> MakeQkvLinearPattern(const QkvCandidateGroup& group) {
    if (group.ambiguous || !group.projections[kQueryProjection].has_value() ||
        !group.projections[kKeyProjection].has_value() ||
        !group.projections[kValueProjection].has_value()) {
        return std::nullopt;
    }

    const auto& query = *group.projections[kQueryProjection];
    const auto& key = *group.projections[kKeyProjection];
    const auto& value = *group.projections[kValueProjection];
    const std::array projections{&query, &key, &value};
    const auto& input_shape = query.input_spec.shape;

    if (!input_shape.IsRanked() || input_shape.rank() == 0U) {
        return std::nullopt;
    }

    const ShapeSymbol& input_features = query.input_spec.shape[*input_shape.rank() - 1U];
    for (const auto* projection: projections) {
        if (projection->weight_spec.dtype != query.weight_spec.dtype ||
            projection->weight_quantization != query.weight_quantization ||
            !AreProvablyEqual(input_features, projection->weight_spec.shape[1])) {
            return std::nullopt;
        }
    }

    if (query.output_features > std::numeric_limits<int64_t>::max() - key.output_features) {
        return std::nullopt;
    }

    const int64_t query_and_key_output_features = query.output_features + key.output_features;
    if (query_and_key_output_features >
        std::numeric_limits<int64_t>::max() - value.output_features) {
        return std::nullopt;
    }
    const int64_t fused_output_features = query_and_key_output_features + value.output_features;

    return QkvLinearPattern{
            .query = ProjectionMatch{
                    .node = query.node,
                    .output = query.output,
                    .output_desc = query.output_desc,
                    .output_features = query.output_features,
            },
            .key = ProjectionMatch{
                    .node = key.node,
                    .output = key.output,
                    .output_desc = key.output_desc,
                    .output_features = key.output_features,
            },
            .value = ProjectionMatch{
                    .node = value.node,
                    .output = value.output,
                    .output_desc = value.output_desc,
                    .output_features = value.output_features,
            },
            .input = group.input,
            .decoder_layer_index = group.decoder_layer_index,
            .fused_weight_spec = TensorSpec{
                    .dtype = query.weight_spec.dtype,
                    .shape = SymbolicShape{
                            ShapeSymbol::CreateFromValue(fused_output_features),
                            query.weight_spec.shape[1],
                    },
            },
            .weight_quantization = query.weight_quantization,
    };
}

NodeOutputDesc MakeOutputDesc(const GraphValueDesc& desc) {
    return NodeOutputDesc{
            .payload = desc.payload,
            .quantization = desc.quantization,
            .name = desc.name,
    };
}

Status FuseQkvLinear(GraphRewriteSession& session, const QkvLinearPattern& pattern) {
    const std::string layer_prefix = "layers." + std::to_string(pattern.decoder_layer_index) +
                                     ".self_attn.qkv_proj";
    const GraphValueId fused_weight = session.AddSessionWeight(
            pattern.fused_weight_spec,
            MakeQkvWeightBinding(pattern.decoder_layer_index),
            pattern.weight_quantization,
            layer_prefix + ".weight");

    SubgraphBuilder builder(
            session,
            {pattern.query.node, pattern.key.node, pattern.value.node});
    AM_ASSIGN_OR_RETURN(
            const std::vector<GraphValueId> fused_outputs,
            builder.Emit(OpType::kQkvLinear,
                         {pattern.input, fused_weight},
                         {MakeOutputDesc(pattern.query.output_desc),
                          MakeOutputDesc(pattern.key.output_desc),
                          MakeOutputDesc(pattern.value.output_desc)},
                         QkvLinearParams{
                                 .q_out_features = pattern.query.output_features,
                                 .k_out_features = pattern.key.output_features,
                                 .v_out_features = pattern.value.output_features,
                                 .has_bias = false,
                         },
                         pattern.decoder_layer_index,
                         layer_prefix));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[kQueryProjection],
                                     pattern.query.output));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[kKeyProjection],
                                     pattern.key.output));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[kValueProjection],
                                     pattern.value.output));
    return builder.Commit();
}

} // namespace

std::string_view QkvLinearFusionPass::Name() const noexcept {
    return "QkvLinearFusionPass";
}

Status QkvLinearFusionPass::Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept {
    if (!ctx.enable_qkv_fusion) {
        return Status::Ok();
    }

    StatusOr<std::vector<GraphNodeId>> topology = session.GetTopologicalOrder();
    AM_RETURN_IF_ERROR(topology.status());

    std::vector<QkvCandidateGroup> groups;
    for (auto node_id: *topology) {
        auto candidate = FindLinearProjectionCandidate(
                session, node_id);
        AM_RETURN_IF_ERROR(candidate.status());
        if (candidate->has_value()) {
            AddToCandidateGroups(groups, std::move(**candidate));
        }
    }

    for (const auto& group: groups) {
        if (const auto pattern = MakeQkvLinearPattern(group); pattern.has_value()) {
            AM_RETURN_IF_ERROR(FuseQkvLinear(session, *pattern));
        }
    }
    return Status::Ok();
}

} // namespace aethermind
