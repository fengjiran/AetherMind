#include "aethermind/model/graph/optimization/silu_mul_fusion_pass.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/op_type.h"

#include <optional>

namespace aethermind {
namespace {

struct SiluMulPattern {
    GraphNodeId silu_node{};
    GraphNodeId mul_node{};
    GraphValueId gate{};
    GraphValueId up{};
    GraphValueId mul_out{};
    std::optional<uint32_t> decoder_layer_index{};
};

StatusOr<std::optional<SiluMulPattern>> FindSiluMulPattern(GraphRewriteSession& session, GraphNodeId silu_node) {
    if (!session.IsNodeLive(silu_node)) {
        return std::optional<SiluMulPattern>{};
    }

    StatusOr<GraphNodeView> silu_view = session.GetNodeView(silu_node);
    AM_RETURN_IF_ERROR(silu_view.status());
    if (silu_view->op_type != OpType::kSilu || silu_view->inputs.size() != 1U ||
        silu_view->outputs.size() != 1U) {
        return std::optional<SiluMulPattern>{};
    }

    const GraphValueId silu_out = silu_view->outputs[0];
    if (!session.IsValueLive(silu_out) || session.IsGraphOutput(silu_out)) {
        return std::optional<SiluMulPattern>{};
    }

    const std::vector<GraphNodeId> consumers = session.FindConsumers(silu_out);
    if (consumers.size() != 1U) {
        return std::optional<SiluMulPattern>{};
    }

    const GraphNodeId mul_node = consumers[0];
    if (!session.IsNodeLive(mul_node)) {
        return std::optional<SiluMulPattern>{};
    }

    StatusOr<GraphNodeView> mul_view = session.GetNodeView(mul_node);
    AM_RETURN_IF_ERROR(mul_view.status());
    if (mul_view->op_type != OpType::kElementwiseMul || mul_view->inputs.size() != 2U ||
        mul_view->outputs.size() != 1U ||
        mul_view->decoder_layer_index != silu_view->decoder_layer_index) {
        return std::optional<SiluMulPattern>{};
    }

    const GraphValueId resolved_silu_out = session.GetResolvedValue(silu_out);
    const GraphValueId first_input = session.GetResolvedValue(mul_view->inputs[0]);
    const GraphValueId second_input = session.GetResolvedValue(mul_view->inputs[1]);
    GraphValueId up;
    if (first_input == resolved_silu_out) {
        up = mul_view->inputs[1];
    } else if (second_input == resolved_silu_out) {
        up = mul_view->inputs[0];
    } else {
        return std::optional<SiluMulPattern>{};
    }

    if (session.GetResolvedValue(up) == resolved_silu_out) {
        return std::optional<SiluMulPattern>{};
    }

    return std::optional<SiluMulPattern>{SiluMulPattern{
            .silu_node = silu_node,
            .mul_node = mul_node,
            .gate = silu_view->inputs[0],
            .up = up,
            .mul_out = mul_view->outputs[0],
            .decoder_layer_index = mul_view->decoder_layer_index,
    }};
}

Status TryFuseSilu(GraphRewriteSession& session, GraphNodeId silu_node) {
    StatusOr<std::optional<SiluMulPattern>> pattern_or = FindSiluMulPattern(session, silu_node);
    AM_RETURN_IF_ERROR(pattern_or.status());
    const std::optional<SiluMulPattern>& pattern = *pattern_or;
    if (!pattern.has_value()) {
        return Status::Ok();
    }

    StatusOr<GraphValueDesc> output_desc = session.GetValueOutputMetadata(pattern->mul_out);
    AM_RETURN_IF_ERROR(output_desc.status());

    SubgraphBuilder builder(session, {pattern->silu_node, pattern->mul_node});
    AM_ASSIGN_OR_RETURN(const GraphValueId fused, builder.Emit(OpType::kSiluMul,
                                                               {pattern->gate, pattern->up},
                                                               NodeOutputDesc{
                                                                       .payload = output_desc->payload,
                                                                       .quantization = output_desc->quantization,
                                                                       .name = output_desc->name,
                                                               },
                                                               SiluMulParams{},
                                                               pattern->decoder_layer_index,
                                                               "silu_mul_fused"));
    AM_RETURN_IF_ERROR(builder.Yield(fused, pattern->mul_out));
    return builder.Commit();
}

}// namespace

std::string_view SiluMulFusionPass::Name() const noexcept {
    return "SiluMulFusionPass";
}

Status SiluMulFusionPass::Run(GraphRewriteSession& session, const PassContext& ctx) {
    if (!ctx.enable_swiglu_fusion) {
        return Status::Ok();
    }

    const std::vector<GraphNodeId> silu_nodes = session.FindNodesByOpType(OpType::kSilu);
    for (GraphNodeId silu_node: silu_nodes) {
        AM_RETURN_IF_ERROR(TryFuseSilu(session, silu_node));
    }
    return Status::Ok();
}

}// namespace aethermind
