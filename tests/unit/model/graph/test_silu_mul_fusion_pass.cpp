#include "aethermind/model/graph/silu_mul_fusion_pass.h"

#include "aethermind/model/graph/graph_op_builder.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace aethermind {
namespace {

TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

NodeOutputDesc ActivationDesc(const char* debug_name) {
    return NodeOutputDesc{.spec = Spec(DataType::Float32(), {2, 4}),
                          .payload = ActivationValue{},
                          .debug_name = debug_name};
}

GraphValueId AddActivation(ModelGraph& graph, const char* debug_name) {
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {2}), std::string(debug_name) + ".tokens");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding},
            std::string(debug_name) + ".weight");
    const AddedNode node = graph.AddNode(OpType::kEmbedding,
                                         std::nullopt,
                                         {tokens, weight},
                                         {ActivationDesc(debug_name)},
                                         EmbeddingParams{},
                                         {},
                                         std::string(debug_name) + ".producer");
    return node.outputs[0];
}

ModelGraph BuildSiluMulPattern(bool reversed_mul_inputs = false) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    const GraphValueId silu = AddSilu(graph, 0U, gate, "silu");
    const GraphValueId output = reversed_mul_inputs
                                        ? AddElementwiseMul(graph, 0U, up, silu, "mul")
                                        : AddElementwiseMul(graph, 0U, silu, up, "mul");
    graph.MarkOutput(output, "output");
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
    EXPECT_EQ(result->GetValue(fused.inputs[0]).debug_name, "gate");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).debug_name, "up");
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
    EXPECT_EQ(result->GetValue(fused.inputs[0]).debug_name, "gate");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).debug_name, "up");
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
    const GraphValueId silu = AddSilu(graph, 0U, gate, "silu");
    const GraphValueId mul = AddElementwiseMul(graph, 0U, silu, up, "mul");
    const GraphValueId add = AddElementwiseAdd(graph, 0U, silu, residual, "add");
    graph.MarkOutput(mul, "mul_output");
    graph.MarkOutput(add, "add_output");

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
    const GraphValueId silu = AddSilu(graph, 0U, gate, "silu");
    const GraphValueId mul = AddElementwiseMul(graph, 0U, silu, up, "mul");
    graph.MarkOutput(silu, "silu_output");
    graph.MarkOutput(mul, "mul_output");

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(SiluMulFusionPass, SkipsMismatchedDecoderLayerIndex) {
    ModelGraph graph;
    const GraphValueId gate = AddActivation(graph, "gate");
    const GraphValueId up = AddActivation(graph, "up");
    const GraphValueId silu = AddSilu(graph, 0U, gate, "silu");
    const GraphValueId mul = AddElementwiseMul(graph, 1U, silu, up, "mul");
    graph.MarkOutput(mul, "output");

    const StatusOr<ModelGraph> result = RunSiluMulFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

}// namespace
}// namespace aethermind
