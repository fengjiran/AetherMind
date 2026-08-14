#include "aethermind/graph/optimization/add_rmsnorm_fusion_pass.h"
#include "aethermind/operators/operator_inference.h"

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {
namespace {

struct AddRmsNormPattern {
    GraphNodeId add_node{};
    GraphNodeId rmsnorm_node{};
    GraphValueId input{};
    GraphValueId residual{};
    GraphValueId weight{};
    GraphValueId add_output{};
    GraphValueId rmsnorm_output{};
    AddRmsNormParams params{};
    GraphValueDesc rmsnorm_output_desc{};
    GraphValueDesc residual_output_desc{};
    std::optional<uint32_t> decoder_layer_index{};
};

using AddProducerIndex = std::unordered_map<uint32_t, GraphNodeId>;

NodeOutputDesc MakeOutputDesc(const GraphValueDesc& desc) {
    return {.payload = desc.payload,
            .quantization = desc.quantization,
            .name = desc.name};
}

bool IsActivationOutput(const GraphValueDesc& desc) noexcept {
    return std::holds_alternative<ActivationValue>(desc.payload);
}

StatusOr<bool> HasObservableUse(const GraphRewriteSession& session, GraphValueId value) {
    if (session.IsGraphOutput(value)) {
        return true;
    }
    return session.HasLiveConsumers(value);
}

StatusOr<AddProducerIndex> BuildAddProducerIndex(const GraphRewriteSession& session) {
    AddProducerIndex producers;
    const auto add_nodes = session.FindNodesByOpType(OpType::kAdd);
    producers.reserve(add_nodes.size());
    for (const auto add_node: add_nodes) {
        const auto add_view = session.GetNodeView(add_node);
        AM_RETURN_IF_ERROR(add_view.status());
        if (add_view->outputs.size() == 1U) {
            producers.emplace(add_view->outputs[0].index, add_node);
        }
    }
    return producers;
}

StatusOr<bool> HasUniqueRmsNormConsumer(const GraphRewriteSession& session,
                                        GraphValueId add_output,
                                        GraphNodeId expected_consumer) {
    AM_ASSIGN_OR_RETURN(const auto consumers,
                        session.FindConsumers(add_output));
    size_t rmsnorm_consumers = 0;
    bool found_expected = false;
    for (const auto consumer: consumers) {
        const auto view = session.GetNodeView(consumer);
        AM_RETURN_IF_ERROR(view.status());
        if (view->op_type != OpType::kRmsNorm || view->inputs.empty() ||
            view->inputs[0] != add_output) {
            continue;
        }
        ++rmsnorm_consumers;
        found_expected |= consumer == expected_consumer;
    }
    return found_expected && rmsnorm_consumers == 1U;
}

StatusOr<std::optional<AddRmsNormPattern>> FindAddRmsNormPattern(
        const GraphRewriteSession& session,
        const AddProducerIndex& add_producers,
        GraphNodeId rmsnorm_node) {
    if (!session.IsNodeLive(rmsnorm_node)) {
        return std::optional<AddRmsNormPattern>{};
    }

    const auto rmsnorm_view = session.GetNodeView(rmsnorm_node);
    AM_RETURN_IF_ERROR(rmsnorm_view.status());
    if (rmsnorm_view->op_type != OpType::kRmsNorm ||
        rmsnorm_view->inputs.size() != 2U ||
        rmsnorm_view->outputs.size() != 1U ||
        !rmsnorm_view->attrs.bytes.empty()) {
        return std::optional<AddRmsNormPattern>{};
    }

    const auto* rmsnorm_params = std::get_if<RmsNormParams>(&rmsnorm_view->op_params);
    if (rmsnorm_params == nullptr) {
        return std::optional<AddRmsNormPattern>{};
    }

    const GraphValueId rmsnorm_output = rmsnorm_view->outputs[0];
    if (!session.IsValueLive(rmsnorm_output) ||
        session.GetResolvedValue(rmsnorm_output) != rmsnorm_output) {
        return std::optional<AddRmsNormPattern>{};
    }

    AM_ASSIGN_OR_RETURN(const bool rmsnorm_has_observable_use,
                        HasObservableUse(session, rmsnorm_output));
    if (!rmsnorm_has_observable_use) {
        return std::optional<AddRmsNormPattern>{};
    }

    const auto producer = add_producers.find(rmsnorm_view->inputs[0].index);
    if (producer == add_producers.end() || !session.IsNodeLive(producer->second)) {
        return std::optional<AddRmsNormPattern>{};
    }

    const GraphNodeId add_node = producer->second;
    const auto add_view = session.GetNodeView(add_node);
    AM_RETURN_IF_ERROR(add_view.status());
    if (add_view->op_type != OpType::kAdd ||
        add_view->inputs.size() != 2U ||
        add_view->outputs.size() != 1U ||
        !std::holds_alternative<AddParams>(add_view->op_params) ||
        !add_view->attrs.bytes.empty()) {
        return std::optional<AddRmsNormPattern>{};
    }

    const GraphValueId add_output = add_view->outputs[0];
    if (rmsnorm_view->inputs[0] != add_output ||
        !session.IsValueLive(add_output) ||
        session.GetResolvedValue(add_output) != add_output) {
        return std::optional<AddRmsNormPattern>{};
    }

    AM_ASSIGN_OR_RETURN(const bool has_unique_rmsnorm_consumer,
                        HasUniqueRmsNormConsumer(session, add_output, rmsnorm_node));
    if (!has_unique_rmsnorm_consumer) {
        return std::optional<AddRmsNormPattern>{};
    }

    StatusOr<GraphValueDesc> input_desc = session.GetValueOutputMetadata(add_view->inputs[1]);
    AM_RETURN_IF_ERROR(input_desc.status());
    StatusOr<GraphValueDesc> residual_desc = session.GetValueOutputMetadata(add_view->inputs[0]);
    AM_RETURN_IF_ERROR(residual_desc.status());
    StatusOr<GraphValueDesc> weight_desc = session.GetValueOutputMetadata(rmsnorm_view->inputs[1]);
    AM_RETURN_IF_ERROR(weight_desc.status());
    StatusOr<GraphValueDesc> add_output_desc = session.GetValueOutputMetadata(add_output);
    AM_RETURN_IF_ERROR(add_output_desc.status());
    StatusOr<GraphValueDesc> rmsnorm_output_desc = session.GetValueOutputMetadata(rmsnorm_output);
    AM_RETURN_IF_ERROR(rmsnorm_output_desc.status());

    if (!IsActivationOutput(*input_desc) || !IsActivationOutput(*residual_desc) ||
        !std::holds_alternative<WeightValue>(weight_desc->payload) ||
        !IsActivationOutput(*add_output_desc) || !IsActivationOutput(*rmsnorm_output_desc)) {
        return std::optional<AddRmsNormPattern>{};
    }

    const AddRmsNormParams fused_params{.eps = rmsnorm_params->eps};
    const std::array<TensorSpec, 3> fused_inputs{
            input_desc->spec,
            residual_desc->spec,
            weight_desc->spec,
    };

    const StatusOr<InferenceResult> inferred = InferOperator(
            OpType::kAddRmsNorm,
            fused_params,
            fused_inputs);
    if (!inferred.ok() || inferred->outputs.size() != 2U ||
        !inferred->runtime_checks.empty() ||
        inferred->outputs[0] != rmsnorm_output_desc->spec ||
        inferred->outputs[1] != add_output_desc->spec) {
        // Rejects broadcast/underdetermined-shape Add and stale or
        // incompatible graph metadata before the rewrite changes any value
        // identity: a fused node with deferred runtime checks would narrow
        // the original Add's broadcast semantics.
        return std::optional<AddRmsNormPattern>{};
    }

    return std::optional{AddRmsNormPattern{
            .add_node = add_node,
            .rmsnorm_node = rmsnorm_node,
            .input = add_view->inputs[1],
            .residual = add_view->inputs[0],
            .weight = rmsnorm_view->inputs[1],
            .add_output = add_output,
            .rmsnorm_output = rmsnorm_output,
            .params = fused_params,
            .rmsnorm_output_desc = std::move(*rmsnorm_output_desc),
            .residual_output_desc = std::move(*add_output_desc),
            .decoder_layer_index = rmsnorm_view->decoder_layer_index}};
}

Status TryFuseAddRmsNorm(GraphRewriteSession& session,
                         const AddProducerIndex& add_producers,
                         GraphNodeId rmsnorm_node) {
    const auto pattern_or = FindAddRmsNormPattern(session, add_producers, rmsnorm_node);
    AM_RETURN_IF_ERROR(pattern_or.status());
    if (!pattern_or->has_value()) {
        return Status::Ok();
    }

    const AddRmsNormPattern& pattern = **pattern_or;
    SubgraphBuilder builder(session, {pattern.add_node, pattern.rmsnorm_node});
    AM_ASSIGN_OR_RETURN(
            const auto fused_outputs,
            builder.Emit(OpType::kAddRmsNorm,
                         {pattern.input, pattern.residual, pattern.weight},
                         {MakeOutputDesc(pattern.rmsnorm_output_desc),
                          MakeOutputDesc(pattern.residual_output_desc)},
                         pattern.params,
                         pattern.decoder_layer_index,
                         "add_rmsnorm_fused"));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[0], pattern.rmsnorm_output));
    AM_RETURN_IF_ERROR(builder.Yield(fused_outputs[1], pattern.add_output));
    return builder.Commit();
}

}// namespace

std::string_view AddRmsNormFusionPass::Name() const noexcept {
    return "AddRmsNormFusionPass";
}

Status AddRmsNormFusionPass::Run(GraphRewriteSession& session,
                                 const PassContext& ctx) const noexcept {
    if (!ctx.enable_fused_add_rms_norm) {
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const auto add_producers,
                        BuildAddProducerIndex(session));
    const auto rmsnorm_nodes = session.FindNodesByOpType(OpType::kRmsNorm);
    for (const auto rmsnorm_node: rmsnorm_nodes) {
        AM_RETURN_IF_ERROR(TryFuseAddRmsNorm(session, add_producers, rmsnorm_node));
    }
    return Status::Ok();
}

}// namespace aethermind
