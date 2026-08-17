#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/graph/optimization/optimize_model_graph.h"
#include "aethermind/graph/optimization/qkv_linear_fusion_pass.h"
#include "test_optimization_helpers.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using namespace test_utils;

constexpr std::array<TransformerWeightRole, 3> kQkvRoles{
        TransformerWeightRole::kAttentionQ,
        TransformerWeightRole::kAttentionK,
        TransformerWeightRole::kAttentionV,
};

struct QkvProjections {
    GraphValueId input{};
    std::array<GraphValueId, 3> outputs{};
};

GraphNodeId Producer(const ModelGraph& graph, GraphValueId value) {
    const std::optional<GraphNodeId> producer = graph.GetValue(value).producer;
    AM_CHECK(producer.has_value(), "expected a producer");
    return *producer;
}

GraphValueId WeightForOutput(const ModelGraph& graph, GraphValueId output) {
    const GraphNode& node = graph.GetNode(Producer(graph, output));
    AM_CHECK(node.inputs.size() == 2U, "expected a two-input linear node");
    return node.inputs[1];
}

StatusOr<GraphValueId> AddNamedProjection(ModelGraph& graph,
                                          GraphValueId input,
                                          int64_t rows,
                                          uint32_t layer,
                                          TransformerWeightRole role,
                                          std::string name) {
    const TensorSpec input_spec = graph.GetValue(input).spec;
    if (!input_spec.shape.IsRanked() || input_spec.shape.rank() == 0U) {
        return Status::InvalidArgument("named projection requires ranked activation input");
    }
    const GraphValueId weight = graph.AddWeight(
            {.dtype = DataType::Float32(),
             .shape = {ShapeSymbol::CreateFromValue(rows),
                       input_spec.shape[*input_spec.shape.rank() - 1U]}},
            MakeTransformerWeightBinding(layer, role),
            name + ".weight");
    StatusOr<AddedNode> node = graph.AddNode(
            OpType::kLinear,
            layer,
            {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}, .name = std::move(name)}},
            LinearParams{});
    if (!node.ok()) {
        return node.status();
    }
    return node->outputs[0];
}

QkvProjections AddQkvProjections(
        ModelGraph& graph,
        uint32_t layer,
        std::array<int64_t, 3> rows = {8, 8, 8},
        std::array<TransformerWeightRole, 3> insertion_order = kQkvRoles,
        std::optional<GraphValueId> input = std::nullopt) {
    if (!input.has_value()) {
        input = AddActivation(graph, "hidden");
    }

    QkvProjections result{.input = *input};
    for (const TransformerWeightRole role: insertion_order) {
        const size_t projection = role == TransformerWeightRole::kAttentionQ
                                          ? 0U
                                  : role == TransformerWeightRole::kAttentionK ? 1U
                                                                               : 2U;
        StatusOr<GraphValueId> output = AddNamedProjection(
                graph,
                *input,
                rows[projection],
                layer,
                role,
                "layers." + std::to_string(layer) + "." +
                        (projection == 0U ? "q_proj" : projection == 1U ? "k_proj"
                                                                        : "v_proj"));
        AM_CHECK(output.ok(), "AddLinear failed: {}", output.status().ToString());
        result.outputs[projection] = *output;
    }
    return result;
}

StatusOr<ModelGraph> RunQkvFusion(const ModelGraph& graph,
                                  PassContext context = {},
                                  bool run_dce = false) {
    GraphPassManager pipeline(context);
    pipeline.Add(std::make_unique<QkvLinearFusionPass>());
    if (run_dce) {
        pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    }
    return pipeline.Run(graph);
}

const GraphNode& OnlyQkvNode(const ModelGraph& graph) {
    const std::vector<GraphNodeId> nodes = graph.FindNodesByOpType(OpType::kQkvLinear);
    AM_CHECK(nodes.size() == 1U, "expected exactly one QkvLinear node");
    return graph.GetNode(nodes[0]);
}

void MarkOutputs(ModelGraph& graph, const QkvProjections& projections) {
    for (const GraphValueId output: projections.outputs) {
        graph.MarkOutput(output);
    }
}

TEST(QkvLinearFusionPass, FusesMhaAndPreservesOutputMetadata) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 3U);
    const QuantizationSpec output_quantization{.kind = QuantizationKind::kInt8,
                                               .group_size = 64U,
                                               .scale_dtype = DataType::Float32(),
                                               .has_zero_point = false};
    graph.SetQuantization(projections.outputs[0], output_quantization);
    MarkOutputs(graph, projections);

    const StatusOr<ModelGraph> result = RunQkvFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(graph.FindNodesByOpType(OpType::kLinear).size(), 3U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 0U);
    const GraphNode& fused = OnlyQkvNode(*result);
    ASSERT_EQ(fused.inputs.size(), 2U);
    ASSERT_EQ(fused.outputs.size(), 3U);
    EXPECT_EQ(fused.decoder_layer_index, 3U);
    EXPECT_EQ(result->GetValue(fused.outputs[0]).name, "layers.3.q_proj");
    EXPECT_EQ(result->GetValue(fused.outputs[0]).quantization, output_quantization);
    const auto* binding = std::get_if<WeightValue>(&result->GetValue(fused.inputs[1]).payload);
    ASSERT_NE(binding, nullptr);
    EXPECT_TRUE(std::holds_alternative<QkvWeightBinding>(binding->binding.spec));
    EXPECT_EQ(binding->binding.decoder_layer_index, 3U);
}

TEST(QkvLinearFusionPass, FusesGqaAndRetainsIndependentKeyValueSplits) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U, {8, 2, 3});
    MarkOutputs(graph, projections);

    const StatusOr<ModelGraph> result = RunQkvFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphNode& fused = OnlyQkvNode(*result);
    const auto* params = std::get_if<QkvLinearParams>(&fused.op_params);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->q_out_features, 8);
    EXPECT_EQ(params->k_out_features, 2);
    EXPECT_EQ(params->v_out_features, 3);
    EXPECT_EQ(result->GetValue(fused.inputs[1]).spec.shape[0].GetStaticValue(), 13);
}

TEST(QkvLinearFusionPass, GroupsByInputLayerAndTopologicalOrderRatherThanInsertionRoleOrder) {
    ModelGraph graph;
    const GraphValueId input = AddActivation(graph, "hidden");
    const QkvProjections layer_one = AddQkvProjections(
            graph,
            1U,
            {9, 3, 2},
            {TransformerWeightRole::kAttentionV,
             TransformerWeightRole::kAttentionQ,
             TransformerWeightRole::kAttentionK},
            input);
    const QkvProjections layer_zero = AddQkvProjections(
            graph,
            0U,
            {8, 4, 4},
            {TransformerWeightRole::kAttentionK,
             TransformerWeightRole::kAttentionV,
             TransformerWeightRole::kAttentionQ},
            input);
    MarkOutputs(graph, layer_one);
    MarkOutputs(graph, layer_zero);

    const StatusOr<ModelGraph> result = RunQkvFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const std::vector<GraphNodeId> fused = result->FindNodesByOpType(OpType::kQkvLinear);
    ASSERT_EQ(fused.size(), 2U);
    const auto* first_params = std::get_if<QkvLinearParams>(&result->GetNode(fused[0]).op_params);
    const auto* second_params = std::get_if<QkvLinearParams>(&result->GetNode(fused[1]).op_params);
    ASSERT_NE(first_params, nullptr);
    ASSERT_NE(second_params, nullptr);
    EXPECT_EQ(result->GetNode(fused[0]).decoder_layer_index, 1U);
    EXPECT_EQ(first_params->q_out_features, 9);
    EXPECT_EQ(result->GetNode(fused[1]).decoder_layer_index, 0U);
    EXPECT_EQ(second_params->q_out_features, 8);
}

TEST(QkvLinearFusionPass, FusesWhenProjectionHasMultipleConsumers) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U);
    StatusOr<GraphValueId> q_consumer = AddLinear(
            graph,
            projections.outputs[0],
            4,
            DataType::Float32(),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionO),
            "layers.0.o_proj");
    ASSERT_TRUE(q_consumer.ok()) << q_consumer.status().ToString();
    graph.MarkOutput(*q_consumer);
    graph.MarkOutput(projections.outputs[1]);
    graph.MarkOutput(projections.outputs[2]);

    const StatusOr<ModelGraph> result = RunQkvFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kQkvLinear).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 1U);
}

TEST(QkvLinearFusionPass, SkipsIncompleteDuplicateAndNonAttentionRoleGroups) {
    ModelGraph graph;
    const GraphValueId input = AddActivation(graph, "hidden");
    const QkvProjections complete = AddQkvProjections(graph, 0U, {8, 8, 8}, kQkvRoles, input);
    StatusOr<GraphValueId> duplicate_query = AddLinear(
            graph,
            input,
            8,
            DataType::Float32(),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            "layers.0.q_proj_duplicate");
    ASSERT_TRUE(duplicate_query.ok()) << duplicate_query.status().ToString();
    StatusOr<GraphValueId> other_role = AddLinear(
            graph,
            input,
            8,
            DataType::Float32(),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionO),
            "layers.0.o_proj");
    ASSERT_TRUE(other_role.ok()) << other_role.status().ToString();
    StatusOr<GraphValueId> generic = AddLinear(graph,
                                               input,
                                               8,
                                               DataType::Float32(),
                                               MakeDirectWeightBinding(ParameterSlot::kKernel),
                                               "generic.kernel");
    ASSERT_TRUE(generic.ok()) << generic.status().ToString();
    const GraphValueId different_input = AddActivation(graph, "different_hidden");
    const auto split_query = AddLinear(
            graph,
            input,
            8,
            DataType::Float32(),
            MakeTransformerWeightBinding(1U, TransformerWeightRole::kAttentionQ),
            "layers.1.q_proj");
    const auto split_key = AddLinear(
            graph,
            input,
            8,
            DataType::Float32(),
            MakeTransformerWeightBinding(1U, TransformerWeightRole::kAttentionK),
            "layers.1.k_proj");
    const auto split_value = AddLinear(
            graph,
            different_input,
            8,
            DataType::Float32(),
            MakeTransformerWeightBinding(1U, TransformerWeightRole::kAttentionV),
            "layers.1.v_proj");
    ASSERT_TRUE(split_query.ok());
    ASSERT_TRUE(split_key.ok());
    ASSERT_TRUE(split_value.ok());
    MarkOutputs(graph, complete);
    graph.MarkOutput(*duplicate_query);
    graph.MarkOutput(*other_role);
    graph.MarkOutput(*generic);
    graph.MarkOutput(*split_query);
    graph.MarkOutput(*split_key);
    graph.MarkOutput(*split_value);

    const StatusOr<ModelGraph> result = RunQkvFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 9U);
}

TEST(QkvLinearFusionPass, SkipsDtypeAndQuantizationMismatches) {
    ModelGraph dtype_graph;
    const QkvProjections dtype_projections = AddQkvProjections(dtype_graph, 0U);
    dtype_graph.SetQuantization(WeightForOutput(dtype_graph, dtype_projections.outputs[0]),
                                {.kind = QuantizationKind::kInt8,
                                 .group_size = 64U,
                                 .scale_dtype = DataType::Float32()});
    MarkOutputs(dtype_graph, dtype_projections);

    const StatusOr<ModelGraph> quantization_result = RunQkvFusion(dtype_graph);

    ASSERT_TRUE(quantization_result.ok()) << quantization_result.status().ToString();
    EXPECT_EQ(quantization_result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);

    ModelGraph role_graph;
    const GraphValueId input = AddActivation(role_graph, "hidden");
    const auto query = AddLinear(role_graph,
                                 input,
                                 8,
                                 DataType::Float32(),
                                 MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
                                 "q");
    const auto key = AddLinear(role_graph,
                               input,
                               8,
                               DataType::Float(16),
                               MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionK),
                               "k");
    const auto value = AddLinear(role_graph,
                                 input,
                                 8,
                                 DataType::Float32(),
                                 MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionV),
                                 "v");
    ASSERT_TRUE(query.ok());
    ASSERT_TRUE(key.ok());
    ASSERT_TRUE(value.ok());
    role_graph.MarkOutput(*query);
    role_graph.MarkOutput(*key);
    role_graph.MarkOutput(*value);

    const StatusOr<ModelGraph> dtype_result = RunQkvFusion(role_graph);

    ASSERT_TRUE(dtype_result.ok()) << dtype_result.status().ToString();
    EXPECT_EQ(dtype_result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);
}

TEST(QkvLinearFusionPass, SkipsSymbolicRowsReplacedOutputsAndDeadOutputs) {
    ModelGraph symbolic_graph;
    const GraphValueId input = AddActivation(symbolic_graph, "hidden");
    const ShapeSymbol input_features = symbolic_graph.GetValue(input).spec.shape[1];
    for (const TransformerWeightRole role: kQkvRoles) {
        const GraphValueId weight = symbolic_graph.AddWeight(
                {.dtype = DataType::Float32(),
                 .shape = {ShapeSymbol::Create(), input_features}},
                MakeTransformerWeightBinding(0U, role),
                "symbolic.weight");
        const StatusOr<AddedNode> node = symbolic_graph.AddNode(
                OpType::kLinear,
                0U,
                {input, weight},
                {NodeOutputDesc{.payload = ActivationValue{}, .name = "symbolic.output"}},
                LinearParams{});
        ASSERT_TRUE(node.ok()) << node.status().ToString();
        symbolic_graph.MarkOutput(node->outputs[0]);
    }

    const StatusOr<ModelGraph> symbolic_result = RunQkvFusion(symbolic_graph);

    ASSERT_TRUE(symbolic_result.ok()) << symbolic_result.status().ToString();
    EXPECT_EQ(symbolic_result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);

    ModelGraph replaced_graph;
    const QkvProjections replaced = AddQkvProjections(replaced_graph, 0U);
    const StatusOr<GraphValueId> alternative = AddLinear(
            replaced_graph,
            replaced.input,
            8,
            DataType::Float32(),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionO),
            "alternative");
    ASSERT_TRUE(alternative.ok()) << alternative.status().ToString();
    MarkOutputs(replaced_graph, replaced);
    GraphRewriteSession session(replaced_graph);
    ASSERT_TRUE(session.ReplaceValue(replaced.outputs[0], *alternative).ok());
    QkvLinearFusionPass pass;
    ASSERT_TRUE(pass.Run(session, {}).ok());
    const StatusOr<ModelGraph> replaced_result = session.Commit();
    ASSERT_TRUE(replaced_result.ok()) << replaced_result.status().ToString();
    EXPECT_EQ(replaced_result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);

    ModelGraph dead_graph;
    (void) AddQkvProjections(dead_graph, 0U);
    const StatusOr<ModelGraph> dead_result = RunQkvFusion(dead_graph);
    ASSERT_TRUE(dead_result.ok()) << dead_result.status().ToString();
    EXPECT_EQ(dead_result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);
}

TEST(QkvLinearFusionPass, SkipsInputFeatureMismatchIntroducedBySessionRedirect) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U);
    MarkOutputs(graph, projections);

    GraphRewriteSession session(graph);
    const GraphValueId incompatible_weight = session.AddSessionWeight(
            Spec(DataType::Float32(), {8, 5}),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kAttentionQ),
            {},
            "incompatible.q_proj.weight");
    ASSERT_TRUE(session.RedirectInput(Producer(graph, projections.outputs[0]), 1U, incompatible_weight)
                        .ok());

    QkvLinearFusionPass pass;
    EXPECT_TRUE(pass.Run(session, {}).ok());

    // A fused replacement would have replaced the mirror node and made this
    // invalid redirect disappear. Its remaining validation failure proves the
    // incompatible Q candidate was skipped instead.
    const StatusOr<ModelGraph> result = session.Commit();
    EXPECT_FALSE(result.ok());
}

TEST(QkvLinearFusionPass, HonorsFlagCheckpointsDceAndIsIdempotent) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U);
    MarkOutputs(graph, projections);

    PassContext disabled;
    disabled.enable_qkv_fusion = false;
    const StatusOr<ModelGraph> disabled_result = RunQkvFusion(graph, disabled);
    ASSERT_TRUE(disabled_result.ok()) << disabled_result.status().ToString();
    EXPECT_EQ(disabled_result->FindNodesByOpType(OpType::kLinear).size(), 3U);

    PassContext checkpointed;
    checkpointed.checkpoint_every = 1U;
    const StatusOr<ModelGraph> result = RunQkvFusion(graph, checkpointed, true);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kQkvLinear).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 0U);

    const StatusOr<ModelGraph> rerun = RunQkvFusion(*result);
    ASSERT_TRUE(rerun.ok()) << rerun.status().ToString();
    EXPECT_EQ(rerun->FindNodesByOpType(OpType::kQkvLinear).size(), 1U);
    EXPECT_EQ(rerun->FindNodesByOpType(OpType::kLinear).size(), 0U);

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(*rerun);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), rerun->GetNodes().size());
}

// ---- Default pipeline integration (OptimizeModelGraph) ---------------------

TEST(QkvLinearFusionPass, DefaultPipelineFusesQkvAtOptLevelTwo) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U);
    MarkOutputs(graph, projections);
    ASSERT_TRUE(graph.Validate().ok());

    // Default PassContext opts into the full O2 pipeline
    // (CF -> QkvLinearFusion -> SiluMulFusion -> DCE).
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kQkvLinear).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 0U);
}

TEST(QkvLinearFusionPass, DefaultPipelineHonorsQkvFusionFlag) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U);
    MarkOutputs(graph, projections);
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.enable_qkv_fusion = false;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 3U);
}

TEST(QkvLinearFusionPass, DefaultPipelineSkipsQkvAtOptLevelOne) {
    ModelGraph graph;
    const QkvProjections projections = AddQkvProjections(graph, 0U);
    MarkOutputs(graph, projections);
    ASSERT_TRUE(graph.Validate().ok());

    // QkvLinearFusion is a semantic fusion pass registered only at O2+;
    // O1 stays a CF -> DCE cleanup pipeline.
    PassContext ctx;
    ctx.opt_level = 1;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kQkvLinear).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 3U);
}

}// namespace
