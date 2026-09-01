#include "aethermind/graph/optimization/gate_up_linear_fusion_pass.h"
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

constexpr size_t kGateProjection = 0U;
constexpr size_t kUpProjection = 1U;
constexpr size_t kProjectionCount = 2U;

struct LinearProjectionCandidate {
    GraphNodeId node{};
    GraphValueId input{};
    GraphValueId output{};
    uint32_t decoder_layer_index = 0;
    TransformerWeightRole role = TransformerWeightRole::kMlpGate;
    TensorSpec input_spec{};
    TensorSpec weight_spec{};
    GraphValueDesc output_desc{};
    QuantizationSpec weight_quantization{};
    int64_t output_features = 0;
};

struct GateUpCandidateGroup {
    GraphValueId input{};
    std::array<std::optional<LinearProjectionCandidate>, kProjectionCount> projections{};
    uint32_t decoder_layer_index = 0;
    bool ambiguous = false;
};

struct ProjectionMatch {
    GraphNodeId node{};
    GraphValueId output{};
    GraphValueDesc output_desc{};
    int64_t output_features = 0;
};

struct GateUpLinearPattern {
    ProjectionMatch gate{};
    ProjectionMatch up{};
    GraphValueId input{};
    uint32_t decoder_layer_index = 0;
    TensorSpec fused_weight_spec{};
    QuantizationSpec weight_quantization{};
};

std::optional<size_t> ProjectionIndex(TransformerWeightRole role) noexcept {
    switch (role) {
        case TransformerWeightRole::kMlpGate:
            return kGateProjection;
        case TransformerWeightRole::kMlpUp:
            return kUpProjection;
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
    const auto inferred = InferOperator(
            OpType::kLinear, node.op_params,
            std::span<const TensorSpec>{inputs});
    return inferred.ok() && inferred->outputs.size() == 1U &&
           inferred->outputs[0] == output.spec;
}

StatusOr<std::optional<LinearProjectionCandidate>> FindLinearProjectionCandidate(
        GraphRewriteSession& session,
        GraphNodeId node_id) {
    if (!session.IsNodeLive(node_id)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    const auto node = session.GetNodeView(node_id);
    AM_RETURN_IF_ERROR(node.status());
    if (!(node->op_type == OpType::kLinear &&
          node->inputs.size() == 2U &&
          node->outputs.size() == 1U &&
          std::holds_alternative<LinearParams>(node->op_params) &&
          node->attrs.bytes.empty() &&
          node->decoder_layer_index.has_value())) {
        return std::optional<LinearProjectionCandidate>{};
    }

    // GetNodeView resolves any prior input redirect or value replacement, so
    // grouping here is by the effective shared activation rather than stale ids.
    const GraphValueId input = node->inputs[0];
    const GraphValueId weight = node->inputs[1];
    const GraphValueId output = node->outputs[0];
    if (session.GetResolvedValue(output) != output ||
        !session.IsValueLive(output)) {
        return std::optional<LinearProjectionCandidate>{};
    }

    const auto has_live_output = session.HasLiveConsumers(output);
    AM_RETURN_IF_ERROR(has_live_output.status());
    if (!session.IsGraphOutput(output) && !*has_live_output) {
        return std::optional<LinearProjectionCandidate>{};
    }

    auto input_desc = session.GetValueOutputMetadata(input);
    AM_RETURN_IF_ERROR(input_desc.status());
    auto weight_desc = session.GetValueOutputMetadata(weight);
    AM_RETURN_IF_ERROR(weight_desc.status());
    auto output_desc = session.GetValueOutputMetadata(output);
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
            .input_spec = std::move(input_desc->spec),
            .weight_spec = std::move(weight_desc->spec),
            .output_desc = std::move(*output_desc),
            .weight_quantization = weight_desc->quantization,
            .output_features = output_features.GetStaticValue(),
    }};
}

void AddToCandidateGroups(std::vector<GateUpCandidateGroup>& groups,
                          LinearProjectionCandidate candidate) {
    GateUpCandidateGroup* group = nullptr;
    for (auto& existing: groups) {
        if (existing.input == candidate.input &&
            existing.decoder_layer_index == candidate.decoder_layer_index) {
            group = &existing;
            break;
        }
    }

    if (group == nullptr) {
        groups.push_back(GateUpCandidateGroup{
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

std::optional<GateUpLinearPattern> MakeGateUpLinearPattern(const GateUpCandidateGroup& group) {
    if (group.ambiguous || !group.projections[kGateProjection].has_value() ||
        !group.projections[kUpProjection].has_value()) {
        return std::nullopt;
    }

    const auto& gate = *group.projections[kGateProjection];
    const auto& up = *group.projections[kUpProjection];
    const std::array projections{&gate, &up};
    const auto& input_shape = gate.input_spec.shape;
    if (!input_shape.IsRanked() || input_shape.rank() == 0U) {
        return std::nullopt;
    }

    const ShapeSymbol& input_features = input_shape[*input_shape.rank() - 1U];
    for (const auto* projection: projections) {
        if (projection->weight_spec.dtype != gate.weight_spec.dtype ||
            projection->weight_quantization != gate.weight_quantization ||
            !AreProvablyEqual(input_features, projection->weight_spec.shape[1])) {
            return std::nullopt;
        }
    }

    if (gate.output_features > std::numeric_limits<int64_t>::max() - up.output_features) {
        return std::nullopt;
    }
    const int64_t fused_output_features = gate.output_features + up.output_features;

    return GateUpLinearPattern{
            .gate = ProjectionMatch{
                    .node = gate.node,
                    .output = gate.output,
                    .output_desc = gate.output_desc,
                    .output_features = gate.output_features,
            },
            .up = ProjectionMatch{
                    .node = up.node,
                    .output = up.output,
                    .output_desc = up.output_desc,
                    .output_features = up.output_features,
            },
            .input = group.input,
            .decoder_layer_index = group.decoder_layer_index,
            .fused_weight_spec = TensorSpec{
                    .dtype = gate.weight_spec.dtype,
                    .shape = SymbolicShape{
                            ShapeSymbol::CreateFromValue(fused_output_features),
                            gate.weight_spec.shape[1],
                    },
            },
            .weight_quantization = gate.weight_quantization,
    };
}

NodeOutputDesc MakeOutputDesc(const GraphValueDesc& desc) {
    return {.payload = desc.payload,
            .quantization = desc.quantization,
            .name = desc.name};
}

Status FuseGateUpLinear(GraphRewriteSession& session, const GateUpLinearPattern& pattern) {
    const std::string layer_prefix = "layers." + std::to_string(pattern.decoder_layer_index) +
                                     ".mlp.gate_up_proj";
    const GraphValueId fused_weight = session.AddSessionWeight(
            pattern.fused_weight_spec,
            MakeGateUpWeightBinding(pattern.decoder_layer_index),
            pattern.weight_quantization,
            layer_prefix + ".weight");

    SubgraphBuilder builder(session, {pattern.gate.node, pattern.up.node});
    AM_ASSIGN_OR_RETURN(
            const std::vector<GraphValueId> fused_outputs,
            builder.Emit(OpType::kGateUpLinear,
                         {pattern.input, fused_weight},
                         {MakeOutputDesc(pattern.gate.output_desc),
                          MakeOutputDesc(pattern.up.output_desc)},
                         GateUpLinearParams{
                                 .gate_out_features = pattern.gate.output_features,
                                 .up_out_features = pattern.up.output_features,
                                 .has_bias = false,
                         },
                         pattern.decoder_layer_index,
                         layer_prefix));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[kGateProjection], pattern.gate.output));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[kUpProjection], pattern.up.output));
    return builder.Commit();
}

} // namespace

std::string_view GateUpLinearFusionPass::Name() const noexcept {
    return "GateUpLinearFusionPass";
}

Status GateUpLinearFusionPass::Run(GraphRewriteSession& session,
                                   const PassContext& ctx) const noexcept {
    if (!ctx.enable_gate_up_fusion) {
        return Status::Ok();
    }

    const auto topology = session.GetTopologicalOrder();
    AM_RETURN_IF_ERROR(topology.status());

    std::vector<GateUpCandidateGroup> groups;
    for (const auto node_id: *topology) {
        auto candidate = FindLinearProjectionCandidate(session, node_id);
        AM_RETURN_IF_ERROR(candidate.status());
        if (candidate->has_value()) {
            AddToCandidateGroups(groups, std::move(**candidate));
        }
    }

    for (const auto& group: groups) {
        if (const auto pattern = MakeGateUpLinearPattern(group); pattern.has_value()) {
            AM_RETURN_IF_ERROR(FuseGateUpLinear(session, *pattern));
        }
    }
    return Status::Ok();
}

} // namespace aethermind
