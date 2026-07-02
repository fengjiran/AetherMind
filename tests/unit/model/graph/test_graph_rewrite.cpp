#include "aethermind/model/graph/graph_rewrite.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace aethermind {
namespace {

TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

ModelGraph BuildTwoEmbeddingGraph() {
    ModelGraph graph;
    const GraphValueId tokens_a = graph.AddInput(Spec(DataType::Int(32), {1, 1}), "tokens_a");
    const GraphValueId tokens_b = graph.AddInput(Spec(DataType::Int(32), {1, 1}), "tokens_b");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    const AddedNode embed_a = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_a, weight},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
                            .payload = ActivationValue{},
                            .debug_name = "hidden_a"}},
            EmbeddingParams{});
    const AddedNode embed_b = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_b, weight},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
                            .payload = ActivationValue{},
                            .debug_name = "hidden_b"}},
            EmbeddingParams{});
    (void) embed_b;
    graph.MarkOutput(embed_a.outputs[0], "output");
    return graph;
}

TEST(GraphRewriteSession, ResolvesChainedValueReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 1}, GraphValueId{.index = 2}).ok());

    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 2});
}

TEST(GraphRewriteSession, ResolvedValueCacheInvalidatesAfterReplaceValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 1});

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 1}, GraphValueId{.index = 2}).ok());
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 2});
}

TEST(GraphRewriteSession, RedirectInputIsVisibleInNodeView) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1}).ok());

    const StatusOr<GraphNodeView> view = session.GetNodeView(GraphNodeId{.index = 0});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    ASSERT_EQ(view->inputs.size(), 2U);
    EXPECT_EQ(view->inputs[0], GraphValueId{.index = 1});
}

TEST(GraphRewriteSession, ReplaceValueAffectsCommit) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    const GraphNode& first_node = committed->GetNode(GraphNodeId{.index = 0});
    ASSERT_EQ(first_node.inputs.size(), 2U);
    EXPECT_EQ(first_node.inputs[0], committed->GetInputs()[1].value);
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, RemovesUnusedNodeOnCommit) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, RejectsInvalidRedirectInput) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const Status status = session.RedirectInput(GraphNodeId{.index = 0}, 99, GraphValueId{.index = 1});

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceNodeWithSingleReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    // Graph layout:
    //   v0=tokens_a, v1=tokens_b, v2=weight
    //   n0=Embedding(v0,v2) → v3=hidden_a
    //   n1=Embedding(v1,v2) → v4=hidden_b
    //   output: v3

    GraphRewriteSession session(graph);

    // Replace n0 with a single Embedding node using different inputs
    GraphNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {GraphValueId{.index = 3}},// references old n0 output for spec/payload
            .op_params = EmbeddingParams{},
            .debug_name = "replacement",
    };
    ASSERT_TRUE(session.ReplaceNode(GraphNodeId{.index = 0}, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).op_type, OpType::kEmbedding);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).inputs[0],
              committed->GetInputs()[1].value);// tokens_b
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceNodeWithMultipleReplacements) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    // Graph layout:
    //   v0=tokens_a, v1=tokens_b, v2=weight
    //   n0=Embedding(v0,v2) → v3=hidden_a
    //   n1=Embedding(v1,v2) → v4=hidden_b
    //   output: v3

    GraphRewriteSession session(graph);

    // Replace n0 with two independent Embedding nodes
    // Both reference v0 and v2 as inputs; the second one's output replaces v3
    GraphNode r1{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {GraphValueId{.index = 4}},// references old v4 for spec (intermediate)
            .op_params = EmbeddingParams{},
            .debug_name = "r1",
    };
    GraphNode r2{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {GraphValueId{.index = 3}},// final output replaces v3
            .op_params = EmbeddingParams{},
            .debug_name = "r2",
    };
    ASSERT_TRUE(session.ReplaceNode(GraphNodeId{.index = 0}, {std::move(r1), std::move(r2)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 3U);// r1 + r2 + n1
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).debug_name, "r1");
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 1}).debug_name, "r2");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceNodeWithEmptyReplacementsActsAsRemove) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceNode(GraphNodeId{.index = 1}, {}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceNodeRejectsInvalidInputId) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    GraphNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 999}},// invalid
            .outputs = {GraphValueId{.index = 3}},
            .op_params = EmbeddingParams{},
            .debug_name = "invalid_replacement",
    };
    const Status status = session.ReplaceNode(GraphNodeId{.index = 0}, {std::move(replacement)});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ApplyBatchOfMixedMutations) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    // Graph layout:
    //   v0=tokens_a, v1=tokens_b, v2=weight
    //   n0=Embedding(v0,v2) → v3=hidden_a
    //   n1=Embedding(v1,v2) → v4=hidden_b
    //   output: v3

    GraphRewriteSession session(graph);

    const std::array<GraphMutation, 2> mutations{
            RemoveNodeCmd{.node = GraphNodeId{.index = 1}},
            RedirectInputCmd{.node = GraphNodeId{.index = 0},
                             .input_index = 0,
                             .new_value = GraphValueId{.index = 1}},
    };

    ASSERT_TRUE(session.Apply(std::span<const GraphMutation>{mutations}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();

    // n1 was removed
    EXPECT_EQ(committed->GetNodes().size(), 1U);

    // n0's first input was redirected from tokens_a(v0) to tokens_b(v1)
    const GraphNode& node = committed->GetNode(GraphNodeId{.index = 0});
    ASSERT_EQ(node.inputs.size(), 2U);
    EXPECT_EQ(node.inputs[0], committed->GetInputs()[1].value);// tokens_b

    EXPECT_TRUE(committed->Validate().ok());
}

// --- Issue L: GetResolvedValue boundary behavior ---

TEST(GraphRewriteSession, GetResolvedValueReturnsIdentityForUnreplacedValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    // No replacements recorded; every value resolves to itself
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 0});
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 4}), GraphValueId{.index = 4});
}

TEST(GraphRewriteSession, GetResolvedValueHandlesOutOfRangeId) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    // Out-of-range ID should return itself, not crash
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 9999}), GraphValueId{.index = 9999});
}

// --- Issue M: Commit() for graph with StateValue ---

ModelGraph BuildGraphWithState() {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {1, 1}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    const GraphValueId k_cache = graph.AddState(
            Spec(DataType::Float32(), {2, 4, 8}),
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
            "kv_cache.layer_0.k");
    (void) k_cache;
    const AddedNode embed = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
                            .payload = ActivationValue{},
                            .debug_name = "hidden"}},
            EmbeddingParams{});
    graph.MarkOutput(embed.outputs[0], "output");
    return graph;
}

TEST(GraphRewriteSession, CommitsGraphWithStateValue) {
    const ModelGraph graph = BuildGraphWithState();
    GraphRewriteSession session(graph);

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_TRUE(committed->Validate().ok());

    // State value should be present in committed graph
    bool found_state = false;
    for (const GraphValue& value: committed->GetValues()) {
        if (std::holds_alternative<StateValue>(value.payload)) {
            found_state = true;
            break;
        }
    }
    EXPECT_TRUE(found_state);
}

// --- Issue N: Commit() for graph with decoder_layer_index ---

TEST(GraphRewriteSession, CommitsGraphPreservingDecoderLayerIndex) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {1, 1}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    const AddedNode embed = graph.AddNode(
            OpType::kEmbedding,
            3U,// decoder_layer_index = 3
            {tokens, weight},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
                            .payload = ActivationValue{},
                            .debug_name = "hidden"}},
            EmbeddingParams{});
    graph.MarkOutput(embed.outputs[0], "output");

    GraphRewriteSession session(graph);
    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());
    EXPECT_TRUE(committed->GetNode(GraphNodeId{.index = 0}).decoder_layer_index.has_value());
    EXPECT_EQ(*committed->GetNode(GraphNodeId{.index = 0}).decoder_layer_index, 3U);
}

// --- Issue O: ValidateEdits() tested separately ---

TEST(GraphRewriteSession, ValidateEditsSucceedsOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    EXPECT_TRUE(session.ValidateEdits().ok());
}

TEST(GraphRewriteSession, ValidateEditsSucceedsAfterValidMutations) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1}).ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 2}).ok());

    EXPECT_TRUE(session.ValidateEdits().ok());
}

// --- Issue H: Commit() rejects monostate external value with specific error ---

TEST(GraphRewriteSession, RejectsMonostateExternalValueOnCommit) {
    // Build a graph with a single monostate external value (no producer).
    // This requires the test-only escape hatch constructor; the public API
    // (AddInput/AddWeight/AddState) never produces monostate values.
    ModelGraph graph(HfModelConfig{},
                     std::vector<GraphNode>{},
                     std::vector<GraphValue>{
                             GraphValue{.payload = std::monostate{},
                                        .spec = Spec(DataType::Float32(), {1}),
                                        .debug_name = "invalid"},
                     });

    GraphRewriteSession session(graph);
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_FALSE(committed.ok());
    EXPECT_EQ(committed.status().code(), StatusCode::kInvalidArgument);
    // Error message must distinguish monostate from a generic payload failure
    EXPECT_NE(committed.status().message().find("monostate"), std::string::npos);
}

// --- ConstantValue: Commit preserves external constant value ---

TEST(GraphRewriteSession, CommitPreservesConstantValue) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {1, 1}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    auto inline_data = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x01}, std::byte{0x02}});
    const GraphValueId constant = graph.AddConstant(
            Spec(DataType::Float32(), {1}),
            ConstantBinding{.name = "scalar.one", .inline_data = std::move(inline_data)},
            "one");
    const AddedNode embed = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
                            .payload = ActivationValue{},
                            .debug_name = "hidden"}},
            EmbeddingParams{});
    graph.MarkOutput(embed.outputs[0], "output");
    (void) constant;

    GraphRewriteSession session(graph);
    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());

    // The constant value should survive Commit with its binding intact.
    bool found_constant = false;
    for (const GraphValue& value: committed->GetValues()) {
        if (const auto* c = std::get_if<ConstantValue>(&value.payload)) {
            found_constant = true;
            EXPECT_EQ(c->binding.name, "scalar.one");
            ASSERT_TRUE(c->binding.inline_data != nullptr);
            EXPECT_EQ(c->binding.inline_data->size(), 2U);
            EXPECT_EQ(value.debug_name, "one");
        }
    }
    EXPECT_TRUE(found_constant);
}

TEST(GraphRewriteSession, CommitPreservesExternalValueQuantization) {
    ModelGraph graph = BuildTwoEmbeddingGraph();
    const QuantizationSpec quantization{.kind = QuantizationKind::kInt4,
                                        .group_size = 32,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = true};
    graph.SetQuantization(GraphValueId{.index = 2}, quantization);

    GraphRewriteSession session(graph);
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());
    bool found_quantized_weight = false;
    for (const GraphValue& value: committed->GetValues()) {
        if (std::holds_alternative<WeightValue>(value.payload)) {
            found_quantized_weight = true;
            EXPECT_EQ(value.quantization, quantization);
        }
    }
    EXPECT_TRUE(found_quantized_weight);
}

TEST(GraphRewriteSession, CommitPreservesNodeOutputQuantization) {
    ModelGraph graph = BuildTwoEmbeddingGraph();
    const QuantizationSpec quantization{.kind = QuantizationKind::kInt8,
                                        .group_size = 64,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = false};
    graph.SetQuantization(GraphValueId{.index = 3}, quantization);

    GraphRewriteSession session(graph);
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());
    ASSERT_EQ(committed->GetOutputs().size(), 1U);
    EXPECT_EQ(committed->GetValue(committed->GetOutputs()[0].value).quantization, quantization);
}

// --- Issue G: ReplaceValue rejects cycles that would make GetResolvedValue
// silently return an arbitrary value along the cycle ---

TEST(GraphRewriteSession, ReplaceValueRejectsDirectCycle) {
    // ReplaceValue(v0, v1) then ReplaceValue(v1, v0) would close a 2-cycle.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());

    const Status status = session.ReplaceValue(GraphValueId{.index = 1}, GraphValueId{.index = 0});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("cycle"), std::string::npos);
}

TEST(GraphRewriteSession, ReplaceValueRejectsIndirectCycle) {
    // Build a chain v0 -> v1 -> v2, then attempt v2 -> v0 to close a 3-cycle.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 1}, GraphValueId{.index = 2}).ok());

    const Status status = session.ReplaceValue(GraphValueId{.index = 2}, GraphValueId{.index = 0});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("cycle"), std::string::npos);
}

TEST(GraphRewriteSession, ReplaceValueAllowsNonCyclicChain) {
    // A linear chain v0 -> v1 -> v2 is fine; GetResolvedValue should walk it.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 1}, GraphValueId{.index = 2}).ok());

    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 2});
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 1}), GraphValueId{.index = 2});
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 2}), GraphValueId{.index = 2});
}

}// namespace
}// namespace aethermind
