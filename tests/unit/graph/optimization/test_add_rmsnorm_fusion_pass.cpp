#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/graph/optimization/add_rmsnorm_fusion_pass.h"
#include "test_optimization_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;
using namespace test_utils;

struct AddRmsNormPair {
    GraphValueId input{};
    GraphValueId residual{};
    GraphValueId add_output{};
    GraphValueId rmsnorm_output{};
};

StatusOr<GraphValueId> AddPlainRmsNorm(ModelGraph& graph,
                                       std::optional<uint32_t> decoder_layer_index,
                                       GraphValueId input,
                                       std::string name) {
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {4}),
            MakeDirectWeightBinding(ParameterSlot::kScale),
            name + ".weight");
    const StatusOr<AddedNode> added = graph.AddNode(
            OpType::kRmsNorm,
            decoder_layer_index,
            {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}, .name = std::move(name)}},
            RmsNormParams{.eps = 2.0e-5F});
    if (!added.ok()) {
        return added.status();
    }
    return added->outputs[0];
}

AddRmsNormPair AddPair(ModelGraph& graph,
                       std::optional<uint32_t> add_layer,
                       std::optional<uint32_t> rmsnorm_layer,
                       const char* prefix) {
    const GraphValueId residual = AddActivation(graph, (std::string(prefix) + ".residual").c_str());
    const GraphValueId input = AddActivation(graph, (std::string(prefix) + ".input").c_str());
    const auto sum = AddElementwiseAdd(
            graph,
            add_layer,
            residual,
            input,
            std::string(prefix) + ".add");
    AM_CHECK(sum.ok(), "{}", sum.status().ToString());
    const auto normalized = AddPlainRmsNorm(
            graph,
            rmsnorm_layer,
            *sum,
            std::string(prefix) + ".rmsnorm");
    AM_CHECK(normalized.ok(), "{}", normalized.status().ToString());
    return {
            .input = input,
            .residual = residual,
            .add_output = *sum,
            .rmsnorm_output = *normalized,
    };
}

StatusOr<ModelGraph> RunAddRmsNormFusion(const ModelGraph& graph,
                                         PassContext context = {}) {
    GraphPassManager pipeline(context);
    pipeline.Add(std::make_unique<AddRmsNormFusionPass>());
    return pipeline.Run(graph);
}

const GraphNode& OnlyNodeWithOp(const ModelGraph& graph, OpType type) {
    const std::vector<GraphNodeId> nodes = graph.FindNodesByOpType(type);
    AM_CHECK(nodes.size() == 1U, "expected one requested node");
    return graph.GetNode(nodes[0]);
}

TEST(AddRmsNormFusionPass, FusesWithDualOutputsMetadataAndRmsNormLayerIndex) {
    ModelGraph graph;
    const AddRmsNormPair pair = AddPair(graph, 0U, 1U, "layer_boundary");
    graph.MarkOutput(pair.rmsnorm_output);
    graph.MarkOutput(pair.add_output);

    const QuantizationSpec norm_quantization{.kind = QuantizationKind::kInt8,
                                             .group_size = 64U,
                                             .scale_dtype = DataType::Float32()};
    graph.SetQuantization(pair.rmsnorm_output, norm_quantization);

    const StatusOr<ModelGraph> result = RunAddRmsNormFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_TRUE(result->FindNodesByOpType(OpType::kAdd).empty());
    EXPECT_TRUE(result->FindNodesByOpType(OpType::kRmsNorm).empty());
    const GraphNode& fused = OnlyNodeWithOp(*result, OpType::kAddRmsNorm);
    ASSERT_EQ(fused.inputs.size(), 3U);
    EXPECT_EQ(result->GetValue(fused.inputs[0]).name, "layer_boundary.input");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).name, "layer_boundary.residual");
    ASSERT_TRUE(fused.decoder_layer_index.has_value());
    EXPECT_EQ(*fused.decoder_layer_index, 1U);
    ASSERT_EQ(fused.outputs.size(), 2U);
    EXPECT_EQ(result->GetValue(fused.outputs[0]).name, "layer_boundary.rmsnorm");
    EXPECT_EQ(result->GetValue(fused.outputs[1]).name,
              graph.GetValue(pair.add_output).name);
    EXPECT_EQ(result->GetValue(fused.outputs[0]).quantization, norm_quantization);
    const auto* params = std::get_if<AddRmsNormParams>(&fused.op_params);
    ASSERT_NE(params, nullptr);
    EXPECT_FLOAT_EQ(params->eps, 2.0e-5F);
}

TEST(AddRmsNormFusionPass, SkipsAddOutputWithMultipleRmsNormConsumers) {
    ModelGraph graph;
    const AddRmsNormPair pair = AddPair(graph, 0U, 1U, "shared_add");
    const StatusOr<GraphValueId> second_norm = AddPlainRmsNorm(
            graph,
            2U,
            pair.add_output,
            "shared_add.second_rmsnorm");
    ASSERT_TRUE(second_norm.ok()) << second_norm.status().ToString();
    graph.MarkOutput(pair.add_output);
    graph.MarkOutput(pair.rmsnorm_output);
    graph.MarkOutput(*second_norm);

    const StatusOr<ModelGraph> result = RunAddRmsNormFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_TRUE(result->FindNodesByOpType(OpType::kAddRmsNorm).empty());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kRmsNorm).size(), 2U);
    EXPECT_EQ(result->GetOutputs().size(), 3U);
}

TEST(AddRmsNormFusionPass, PreservesExtraNonRmsNormResidualConsumer) {
    ModelGraph graph;
    const AddRmsNormPair pair = AddPair(graph, 0U, 1U, "shared_residual");
    const GraphValueId other = AddActivation(graph, "shared_residual.other");
    const StatusOr<GraphValueId> extra = AddElementwiseAdd(
            graph, 1U, pair.add_output, other, "shared_residual.extra_add");
    ASSERT_TRUE(extra.ok()) << extra.status().ToString();
    graph.MarkOutput(pair.add_output);
    graph.MarkOutput(pair.rmsnorm_output);
    graph.MarkOutput(*extra);

    const StatusOr<ModelGraph> result = RunAddRmsNormFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const GraphNode& fused = OnlyNodeWithOp(*result, OpType::kAddRmsNorm);
    const std::vector<GraphNodeId> adds = result->FindNodesByOpType(OpType::kAdd);
    ASSERT_EQ(adds.size(), 1U);
    EXPECT_EQ(result->GetNode(adds[0]).inputs[0], fused.outputs[1]);
    EXPECT_EQ(result->GetOutputs().size(), 3U);
}

TEST(AddRmsNormFusionPass, FusesLayerInternalCrossLayerAndFinalNormPatterns) {
    ModelGraph graph;
    const AddRmsNormPair layer_internal = AddPair(graph, 0U, 0U, "layer0.attention");
    const AddRmsNormPair cross_layer = AddPair(graph, 0U, 1U, "layer0.mlp");
    const AddRmsNormPair final_norm = AddPair(graph, 1U, std::nullopt, "layer1.mlp");
    graph.MarkOutput(layer_internal.rmsnorm_output);
    graph.MarkOutput(cross_layer.rmsnorm_output);
    graph.MarkOutput(final_norm.rmsnorm_output);

    const StatusOr<ModelGraph> result = RunAddRmsNormFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAddRmsNorm).size(), 3U);
    EXPECT_TRUE(result->FindNodesByOpType(OpType::kAdd).empty());
    EXPECT_TRUE(result->FindNodesByOpType(OpType::kRmsNorm).empty());
    const auto fused = result->FindNodesByOpType(OpType::kAddRmsNorm);
    ASSERT_EQ(fused.size(), 3U);
    EXPECT_EQ(result->GetNode(fused[0]).decoder_layer_index, 0U);
    EXPECT_EQ(result->GetNode(fused[1]).decoder_layer_index, 1U);
    EXPECT_FALSE(result->GetNode(fused[2]).decoder_layer_index.has_value());
}

TEST(AddRmsNormFusionPass, SkipsBroadcastReplacedAndDeadPatterns) {
    ModelGraph broadcast_graph;
    const GraphValueId activation = AddActivation(broadcast_graph, "broadcast.activation");
    const GraphValueId bias = AddFloatConstant(
            broadcast_graph,
            {0.0F, 0.0F, 0.0F, 0.0F},
            {4},
            "broadcast.bias");
    const StatusOr<GraphValueId> broadcast_add = AddElementwiseAdd(
            broadcast_graph, 0U, activation, bias, "broadcast.add");
    ASSERT_TRUE(broadcast_add.ok()) << broadcast_add.status().ToString();
    const StatusOr<GraphValueId> broadcast_norm = AddPlainRmsNorm(
            broadcast_graph, 0U, *broadcast_add, "broadcast.rmsnorm");
    ASSERT_TRUE(broadcast_norm.ok()) << broadcast_norm.status().ToString();
    broadcast_graph.MarkOutput(*broadcast_norm);
    const StatusOr<ModelGraph> broadcast_result = RunAddRmsNormFusion(broadcast_graph);
    ASSERT_TRUE(broadcast_result.ok()) << broadcast_result.status().ToString();
    EXPECT_TRUE(broadcast_result->FindNodesByOpType(OpType::kAddRmsNorm).empty());

    ModelGraph replaced_graph;
    const AddRmsNormPair replaced = AddPair(replaced_graph, 0U, 0U, "replaced");
    const GraphValueId alternative = AddActivation(replaced_graph, "replaced.alternative");
    replaced_graph.MarkOutput(replaced.rmsnorm_output);
    GraphRewriteSession session(replaced_graph);
    ASSERT_TRUE(session.ReplaceValue(replaced.rmsnorm_output, alternative).ok());
    AddRmsNormFusionPass pass;
    ASSERT_TRUE(pass.Run(session, {}).ok());
    const StatusOr<ModelGraph> replaced_result = session.Commit();
    ASSERT_TRUE(replaced_result.ok()) << replaced_result.status().ToString();
    EXPECT_TRUE(replaced_result->FindNodesByOpType(OpType::kAddRmsNorm).empty());

    ModelGraph dead_graph;
    const AddRmsNormPair dead = AddPair(dead_graph, 0U, 0U, "dead");
    dead_graph.MarkOutput(dead.input);
    const StatusOr<ModelGraph> dead_result = RunAddRmsNormFusion(dead_graph);
    ASSERT_TRUE(dead_result.ok()) << dead_result.status().ToString();
    EXPECT_TRUE(dead_result->FindNodesByOpType(OpType::kAddRmsNorm).empty());
}

TEST(AddRmsNormFusionPass, SkipsSymbolicResidualBroadcastCompatibleWithStaticInput) {
    // Regression: input {2,4} + residual {2,sym} infers Add output {2,4}
    // (static-priority broadcast) and a fused AddRmsNorm output {2,4} as
    // well, so output-spec equality alone would accept the pattern while the
    // fused op's deferred exact-shape runtime check narrows the original
    // Add's broadcast semantics. The pass must skip whenever the fused op
    // carries any runtime check.
    ModelGraph graph;
    const ShapeSymbol residual_hidden = ShapeSymbol::Create();
    const GraphValueId input = AddFloatConstant(
            graph,
            {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F},
            {2, 4},
            "symbolic.input");
    const GraphValueId residual = graph.AddConstant(
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape({ShapeSymbol::CreateFromValue(2),
                                               residual_hidden})},
            ConstantBinding{},
            "symbolic.residual");
    const StatusOr<GraphValueId> add = AddElementwiseAdd(
            graph, 0U, residual, input, "symbolic.add");
    ASSERT_TRUE(add.ok()) << add.status().ToString();
    const StatusOr<GraphValueId> norm = AddPlainRmsNorm(
            graph, 0U, *add, "symbolic.rmsnorm");
    ASSERT_TRUE(norm.ok()) << norm.status().ToString();
    graph.MarkOutput(*norm);

    const StatusOr<ModelGraph> result = RunAddRmsNormFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_TRUE(result->FindNodesByOpType(OpType::kAddRmsNorm).empty());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kRmsNorm).size(), 1U);
}

TEST(AddRmsNormFusionPass, HonorsFlagCheckpointsAndIsIdempotent) {
    ModelGraph graph;
    const AddRmsNormPair pair = AddPair(graph, 0U, 0U, "idempotent");
    graph.MarkOutput(pair.rmsnorm_output);

    PassContext disabled;
    disabled.enable_fused_add_rms_norm = false;
    const StatusOr<ModelGraph> disabled_result = RunAddRmsNormFusion(graph, disabled);
    ASSERT_TRUE(disabled_result.ok()) << disabled_result.status().ToString();
    EXPECT_TRUE(disabled_result->FindNodesByOpType(OpType::kAddRmsNorm).empty());

    PassContext checkpointed;
    checkpointed.checkpoint_every = 1U;
    const StatusOr<ModelGraph> first = RunAddRmsNormFusion(graph, checkpointed);
    ASSERT_TRUE(first.ok()) << first.status().ToString();
    ASSERT_TRUE(first->Validate().ok());
    EXPECT_EQ(first->FindNodesByOpType(OpType::kAddRmsNorm).size(), 1U);

    const StatusOr<ModelGraph> second = RunAddRmsNormFusion(*first);
    ASSERT_TRUE(second.ok()) << second.status().ToString();
    EXPECT_EQ(second->FindNodesByOpType(OpType::kAddRmsNorm).size(), 1U);
}

} // namespace
