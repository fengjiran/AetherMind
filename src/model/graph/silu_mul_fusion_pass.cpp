#include "aethermind/model/graph/silu_mul_fusion_pass.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/op_type.h"

namespace aethermind {
namespace {

Status TryFuseSilu(GraphRewriteSession& session, GraphNodeId silu_node) {
    if (!session.IsNodeLive(silu_node)) {
        return Status::Ok();
    }

    StatusOr<GraphNodeView> silu_view = session.GetNodeView(silu_node);
    AM_RETURN_IF_ERROR(silu_view.status());
    if (silu_view->op_type != OpType::kSilu || silu_view->inputs.size() != 1U ||
        silu_view->outputs.size() != 1U) {
        return Status::Ok();
    }

    const GraphValueId silu_out = silu_view->outputs[0];
    if (!session.IsValueLive(silu_out) || session.IsGraphOutput(silu_out)) {
        return Status::Ok();
    }

    const std::vector<GraphNodeId> consumers = session.FindConsumers(silu_out);
    if (consumers.size() != 1U) {
        return Status::Ok();
    }

    const GraphNodeId mul_node = consumers[0];
    if (!session.IsNodeLive(mul_node)) {
        return Status::Ok();
    }

    StatusOr<GraphNodeView> mul_view = session.GetNodeView(mul_node);
    AM_RETURN_IF_ERROR(mul_view.status());
    if (mul_view->op_type != OpType::kElementwiseMul || mul_view->inputs.size() != 2U ||
        mul_view->outputs.size() != 1U ||
        mul_view->decoder_layer_index != silu_view->decoder_layer_index) {
        return Status::Ok();
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
        return Status::Ok();
    }

    const GraphValueId mul_out = mul_view->outputs[0];
    StatusOr<NodeOutputDesc> output_desc = session.GetValueOutputDesc(mul_out);
    AM_RETURN_IF_ERROR(output_desc.status());

    SubgraphBuilder builder(session, {silu_node, mul_node});
    const GraphValueId fused = builder.Emit(OpType::kSiluMul,
                                            {silu_view->inputs[0], up},
                                            *output_desc,
                                            SiluMulParams{},
                                            mul_view->decoder_layer_index,
                                            "silu_mul_fused");
    AM_RETURN_IF_ERROR(builder.Yield(fused, mul_out));
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
