#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/graph/optimization/constant_folding_pass.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/graph/optimization/silu_mul_fusion_pass.h"
#include "test_optimization_helpers.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

using namespace aethermind;

using namespace test_utils;

ModelGraph BuildSiluMulPattern(bool reversed_mul_inputs = false) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    AM_CHECK(silu_or.ok(), "{}", silu_or.status().ToString());
    const GraphValueId silu = *silu_or;
    auto output_or = reversed_mul_inputs
                             ? AddElementwiseMul(graph, 0U, up, silu, "mul")
                             : AddElementwiseMul(graph, 0U, silu, up, "mul");
    AM_CHECK(output_or.ok(), "{}", output_or.status().ToString());
    const GraphValueId output = *output_or;
    graph.MarkOutput(output);
    return graph;
}

StatusOr<ModelGraph> RunSiluMulFusion(const ModelGraph& graph, PassContext ctx = {}) {
    GraphPassManager pipeline(ctx);
    pipeline.Add(std::make_unique<SiluMulFusionPass>());
    return pipeline.Run(graph);
}

const GraphNode& OnlyNodeWithOp(const ModelGraph& graph, OpType op_type) {
    const std::vector<GraphNodeId> nodes = graph.FindNodesByOpType(op_type);
    AM_CHECK(nodes.size() == 1U, "Expected exactly one node with requested op type");
    return graph.GetNode(nodes[0]);
}

TEST(SiluMulFusionPass, FusesSiluMulPattern) {
    ModelGraph graph = BuildSiluMulPattern();
    const QuantizationSpec quantization{.kind = QuantizationKind::kInt8,
                                        .group_size = 64,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = false};
    ASSERT_EQ(graph.GetOutputs().size(), 1U);
    graph.SetQuantization(graph.GetOutputs()[0].value, quantization);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    const GraphNode& fused = OnlyNodeWithOp(*result, OpType::kSiluMul);
    ASSERT_EQ(fused.inputs.size(), 2U);
    EXPECT_EQ(result->GetValue(fused.inputs[0]).name, "gate");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).name, "up");
    ASSERT_EQ(fused.outputs.size(), 1U);
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    EXPECT_EQ(result->GetOutputs()[0].value, fused.outputs[0]);
    EXPECT_EQ(result->GetValue(fused.outputs[0]).quantization, quantization);
    EXPECT_TRUE(std::holds_alternative<SiluMulParams>(fused.op_params));
}

TEST(SiluMulFusionPass, FusesReversedMulInputs) {
    const ModelGraph graph = BuildSiluMulPattern(true);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphNode& fused = OnlyNodeWithOp(*result, OpType::kSiluMul);
    ASSERT_EQ(fused.inputs.size(), 2U);
    EXPECT_EQ(result->GetValue(fused.inputs[0]).name, "gate");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).name, "up");
}

TEST(SiluMulFusionPass, SkipsWhenFusionDisabled) {
    const ModelGraph graph = BuildSiluMulPattern();
    PassContext ctx;
    ctx.enable_swiglu_fusion = false;

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(SiluMulFusionPass, SkipsSiluOutputWithMultipleConsumers) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    const GraphValueId residual = AddActivation(graph, "residual");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    const GraphValueId silu = *silu_or;
    auto mul_or = AddElementwiseMul(graph, 0U, silu, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    const GraphValueId mul = *mul_or;
    auto add_or = AddElementwiseAdd(graph, 0U, silu, residual, "add");
    ASSERT_TRUE(add_or.ok()) << add_or.status().ToString();
    const GraphValueId add = *add_or;
    graph.MarkOutput(mul);
    graph.MarkOutput(add);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(SiluMulFusionPass, SkipsGraphOutputSiluIntermediate) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    const GraphValueId silu = *silu_or;
    auto mul_or = AddElementwiseMul(graph, 0U, silu, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    const GraphValueId mul = *mul_or;
    graph.MarkOutput(silu);
    graph.MarkOutput(mul);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(SiluMulFusionPass, SkipsFullyDeadSiluMulChain) {
    // mul_out is neither a graph output nor consumed by any node. Fusing
    // would produce a rewrite replacement node that DeadCodeEliminationPass
    // cannot traverse (source-graph nodes only), leaking an orphan fused
    // kernel into the committed graph; the chain must stay unfused so DCE
    // can remove the plain Silu and Mul.
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    auto mul_or = AddElementwiseMul(graph, 0U, *silu_or, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    graph.MarkOutput(up);

    const StatusOr<ModelGraph> fused = RunSiluMulFusion(graph);

    ASSERT_TRUE(fused.ok()) << fused.status().ToString();
    EXPECT_EQ(fused->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    EXPECT_EQ(fused->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(fused->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);

    // With DCE behind fusion the fully-dead chain disappears entirely.
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<SiluMulFusionPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
}

TEST(SiluMulFusionPass, SkipsMismatchedDecoderLayerIndex) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    const GraphValueId silu = *silu_or;
    auto mul_or = AddElementwiseMul(graph, 1U, silu, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    const GraphValueId mul = *mul_or;
    graph.MarkOutput(mul);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(SiluMulFusionPass, SkipsWhenMulInputsBothSiluOutput) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    const GraphValueId silu = *silu_or;
    auto mul_or = AddElementwiseMul(graph, 0U, silu, silu, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    const GraphValueId mul = *mul_or;
    graph.MarkOutput(mul);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(SiluMulFusionPass, FusesMultipleSiluMulPairs) {
    ModelGraph graph;
    const GraphValueId gate_a = AddActivation(graph, "gate_a");
    const GraphValueId up_a = AddActivation(graph, "up_a");
    auto silu_a_or = AddSilu(graph, 0U, gate_a, "silu_a");
    ASSERT_TRUE(silu_a_or.ok()) << silu_a_or.status().ToString();
    const GraphValueId silu_a = *silu_a_or;
    auto mul_a_or = AddElementwiseMul(graph, 0U, silu_a, up_a, "mul_a");
    ASSERT_TRUE(mul_a_or.ok()) << mul_a_or.status().ToString();
    const GraphValueId mul_a = *mul_a_or;

    const GraphValueId gate_b = AddActivation(graph, "gate_b");
    const GraphValueId up_b = AddActivation(graph, "up_b");
    auto silu_b_or = AddSilu(graph, 1U, gate_b, "silu_b");
    ASSERT_TRUE(silu_b_or.ok()) << silu_b_or.status().ToString();
    const GraphValueId silu_b = *silu_b_or;
    auto mul_b_or = AddElementwiseMul(graph, 1U, silu_b, up_b, "mul_b");
    ASSERT_TRUE(mul_b_or.ok()) << mul_b_or.status().ToString();
    const GraphValueId mul_b = *mul_b_or;

    graph.MarkOutput(mul_a);
    graph.MarkOutput(mul_b);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 2U);
}

TEST(SiluMulFusionPass, ReplaceValueOnGateInputResolvesAtCommit) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    const GraphValueId alt = AddActivation(graph, "alt");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    const GraphValueId silu = *silu_or;
    auto mul_or = AddElementwiseMul(graph, 0U, silu, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    const GraphValueId mul = *mul_or;
    graph.MarkOutput(mul);

    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.ReplaceValue(gate, alt).ok());

    SiluMulFusionPass pass;
    ASSERT_TRUE(pass.Run(session, {}).ok());

    const StatusOr<ModelGraph> result = session.Commit();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);

    const GraphNode& fused = OnlyNodeWithOp(*result, OpType::kSiluMul);
    ASSERT_EQ(fused.inputs.size(), 2U);
    EXPECT_EQ(result->GetValue(fused.inputs[0]).name, "alt");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).name, "up");
}

TEST(SiluMulFusionPass, ReplaceValueOnUpInputResolvesAtCommit) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    const GraphValueId silu = *silu_or;
    auto mul_or = AddElementwiseMul(graph, 0U, silu, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    const GraphValueId mul = *mul_or;
    graph.MarkOutput(mul);

    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.ReplaceValue(up, gate).ok());

    SiluMulFusionPass pass;
    ASSERT_TRUE(pass.Run(session, {}).ok());

    const StatusOr<ModelGraph> result = session.Commit();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);

    const GraphNode& fused = OnlyNodeWithOp(*result, OpType::kSiluMul);
    ASSERT_EQ(fused.inputs.size(), 2U);
    EXPECT_EQ(result->GetValue(fused.inputs[0]).name, "gate");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).name, "gate");
}

TEST(SiluMulFusionPass, NoSiluNodesLeavesGraphUnchanged) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    graph.MarkOutput(gate);

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
}

// Full O2 pipeline (CF -> Fusion -> DCE) on Silu(constant) * up. Constant
// folding collapses silu(gate) into a folded constant C, so the fusion pass
// must NOT rebuild the silu computation as a fused kernel: the ideal result
// is a plain Mul(C, up) with no runtime silu. If fusion fires anyway, the
// fused kernel takes the raw constant gate as input, re-evaluating silu at
// runtime, and C becomes a dead constant.

TEST(SiluMulFusionPass, DoesNotReintroduceFoldedSiluAsFusedKernel) {
    ModelGraph graph;
    // Match the Float32[2,4] spec of AddActivation's Embedding output so the
    // Mul schema validation passes.
    const GraphValueId gate = AddFloatConstant(
            graph, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}, {2, 4}, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    auto silu_or = AddSilu(graph, 0U, gate, "silu");
    ASSERT_TRUE(silu_or.ok()) << silu_or.status().ToString();
    auto mul_or = AddElementwiseMul(graph, 0U, *silu_or, up, "mul");
    ASSERT_TRUE(mul_or.ok()) << mul_or.status().ToString();
    graph.MarkOutput(*mul_or);

    PassContext ctx;
    GraphPassManager pipeline(ctx);
    pipeline.Add(std::make_unique<ConstantFoldingPass>());
    pipeline.Add(std::make_unique<SiluMulFusionPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    // No fused kernel: silu is already folded away.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    // A single Mul consumes the folded constant.
    const std::vector<GraphNodeId> muls = result->FindNodesByOpType(OpType::kElementwiseMul);
    ASSERT_EQ(muls.size(), 1U);
    const GraphNode& mul = result->GetNode(muls[0]);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(result->GetValue(mul.inputs[0]).payload));
    // Values: gate, up.tokens, up.weight, up, folded C, mul output. The
    // folded constant survives commit because Mul still consumes it.
    EXPECT_EQ(result->GetValues().size(), 6U);
    size_t constant_count = 0;
    for (const auto& value: result->GetValues()) {
        if (std::holds_alternative<ConstantValue>(value.payload)) {
            ++constant_count;
        }
    }
    EXPECT_EQ(constant_count, 2U);
}

} // namespace
