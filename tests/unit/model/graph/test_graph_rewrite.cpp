#include "aethermind/model/graph/graph_rewrite.h"

#include <gtest/gtest.h>

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
                                                WeightBinding{.role = WeightRole::kTokenEmbedding},
                                                "embed.weight");
    const ModelGraph::AddedNode embed_a = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_a, weight},
            {ModelGraph::NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
                                        .payload = ActivationValue{},
                                        .debug_name = "hidden_a"}},
            EmbeddingParams{});
    const ModelGraph::AddedNode embed_b = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_b, weight},
            {ModelGraph::NodeOutputDesc{.spec = Spec(DataType::Float32(), {1, 1, 4}),
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

TEST(GraphRewriteSession, RedirectInputIsVisibleInNodeView) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1}).ok());

    const StatusOr<GraphNodeView> view = session.GetNodeView(GraphNodeId{.index = 0});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    ASSERT_EQ(view->inputs.size(), 2U);
    EXPECT_EQ(view->inputs[0], GraphValueId{.index = 1});
}

TEST(GraphRewriteSession, ReplaceAllUsesAffectsCommit) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceAllUses(GraphValueId{.index = 0}, GraphValueId{.index = 1}).ok());
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
    };
    const Status status = session.ReplaceNode(GraphNodeId{.index = 0}, {std::move(replacement)});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
