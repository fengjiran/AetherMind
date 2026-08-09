#include "aethermind/graph/optimization/silu_mul_fusion_pass.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"

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

    // An earlier pass (e.g. constant folding) may have replaced silu_out with
    // another value. The silu computation no longer exists at runtime, so
    // fusing it back would re-introduce the eliminated work as a fused kernel
    // and orphan the replacement value (e.g. a folded constant). Skip: the
    // plain Mul on the replacement is already optimal.
    if (session.GetResolvedValue(silu_out) != silu_out) {
        return std::optional<SiluMulPattern>{};
    }

    // FindConsumers can fail when the source graph contains a cycle; the
    // fusion pass runs on the unmodified graph, so treat that as "no pattern"
    // only after reporting the error to the caller.
    StatusOr<std::vector<GraphNodeId>> consumers_or = session.FindConsumers(silu_out);
    AM_RETURN_IF_ERROR(consumers_or.status());
    const auto& consumers = *consumers_or;
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

    const GraphValueId mul_out = mul_view->outputs[0];
    // A replacement from an earlier pass (e.g. constant folding) means the
    // Mul no longer exists at runtime; fusing it back would re-introduce the
    // eliminated work and conflict with the replacement binding.
    if (session.GetResolvedValue(mul_out) != mul_out) {
        return std::optional<SiluMulPattern>{};
    }

    // A fully-dead chain (Mul output unreferenced and not a graph output)
    // must not be fused: the fused node is a rewrite replacement, invisible
    // to DeadCodeEliminationPass (which only traverses source-graph nodes),
    // so it would leak an orphan fused kernel into the committed graph.
    if (!session.IsGraphOutput(mul_out)) {
        StatusOr<bool> has_consumers_or = session.HasLiveConsumers(mul_out);
        AM_RETURN_IF_ERROR(has_consumers_or.status());
        if (!*has_consumers_or) {
            return std::optional<SiluMulPattern>{};
        }
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
            .mul_out = mul_out,
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
    AM_ASSIGN_OR_RETURN(const GraphValueId fused,
                        builder.Emit(OpType::kSiluMul,
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

Status SiluMulFusionPass::Run(GraphRewriteSession& session, const PassContext& ctx) const noexcept {
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
