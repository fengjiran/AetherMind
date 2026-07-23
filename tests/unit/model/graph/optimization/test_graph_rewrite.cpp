#include "../test_graph_helpers.h"
#include "aethermind/model/graph/optimization/graph_rewrite.h"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace aethermind {
namespace {

NodeOutputDesc HiddenDesc(const char* debug_name) {
    return NodeOutputDesc{.payload = ActivationValue{},
                          .name = debug_name};
}

RewriteOutputBinding ReplacesHidden(GraphValueId value, const char* debug_name) {
    return RewriteOutputBinding{.desc = HiddenDesc(debug_name), .replaces = value};
}

ModelGraph BuildTwoEmbeddingGraph() {
    ModelGraph graph;
    const GraphValueId tokens_a = graph.AddInput(
            Spec(DataType::Int(32), {1}), "tokens_a");
    const GraphValueId tokens_b = graph.AddInput(
            Spec(DataType::Int(32), {1}), "tokens_b");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding},
            "embed.weight");
    auto embed_a_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_a, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden_a"}},
            EmbeddingParams{});
    AM_CHECK(embed_a_or.ok(), "BuildTwoEmbeddingGraph embed_a AddNode failed");
    const AddedNode& embed_a = *embed_a_or;
    auto embed_b_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_b, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden_b"}},
            EmbeddingParams{});
    AM_CHECK(embed_b_or.ok(), "BuildTwoEmbeddingGraph embed_b AddNode failed");
    const AddedNode& embed_b = *embed_b_or;
    UNUSED(embed_b);
    graph.MarkOutput(embed_a.outputs[0]);
    return graph;
}

RoPEParams ValidRoPEParams() {
    return RoPEParams{.head_dim = 4,
                      .num_attention_heads = 1,
                      .num_key_value_heads = 1,
                      .max_position_embeddings = 128,
                      .theta = 10000.0,
                      .scaling_type = HfRopeScalingType::kNone};
}

ModelGraph BuildRoPEGraph() {
    ModelGraph graph;
    const GraphValueId tokens_a = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens_a");
    const GraphValueId tokens_b = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens_b");
    const GraphValueId position_ids = graph.AddInput(Spec(DataType::Int(64), {1}), "position_ids");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding},
            "embed.weight");
    auto q_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_a, weight},
            {HiddenDesc("q")},
            EmbeddingParams{});
    AM_CHECK(q_or.ok(), "BuildRoPEGraph q AddNode failed");
    const AddedNode& q = *q_or;
    auto k_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_b, weight},
            {HiddenDesc("k")},
            EmbeddingParams{});
    AM_CHECK(k_or.ok(), "BuildRoPEGraph k AddNode failed");
    const AddedNode& k = *k_or;
    auto rope_or = graph.AddNode(
            OpType::kRoPE,
            0U,
            {q.outputs[0], k.outputs[0], position_ids},
            {HiddenDesc("q_rope"), HiddenDesc("k_rope")},
            ValidRoPEParams());
    AM_CHECK(rope_or.ok(), "BuildRoPEGraph rope AddNode failed");
    const AddedNode& rope = *rope_or;
    graph.MarkOutput(rope.outputs[0]);
    graph.MarkOutput(rope.outputs[1]);
    return graph;
}

TEST(GraphRewriteSession, ResolvesChainedValueReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    EXPECT_EQ(graph.GetValues().size(), 5);
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0},
                                     GraphValueId{.index = 1})
                        .ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 1},
                                     GraphValueId{.index = 2})
                        .ok());

    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 2});
}

TEST(GraphRewriteSession, ResolvedValueCacheInvalidatesAfterReplaceValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0},
                                     GraphValueId{.index = 1})
                        .ok());
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 1});

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 1},
                                     GraphValueId{.index = 2})
                        .ok());
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 0}), GraphValueId{.index = 2});
}

TEST(GraphRewriteSession, RedirectInputIsVisibleInNodeView) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0,
                                      GraphValueId{.index = 1})
                        .ok());

    const StatusOr<GraphNodeView> view = session.GetNodeView(GraphNodeId{.index = 0});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    ASSERT_EQ(view->inputs.size(), 2U);
    EXPECT_EQ(view->inputs[0], GraphValueId{.index = 1});
}

TEST(GraphRewriteSession, ReplaceValueAffectsCommit) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 0},
                                     GraphValueId{.index = 1})
                        .ok());
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

TEST(GraphRewriteSession, ReplaceSubgraphWithSingleReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    // Graph layout:
    //   v0=tokens_a, v1=tokens_b, v2=weight
    //   n0=Embedding(v0,v2) → v3=hidden_a
    //   n1=Embedding(v1,v2) → v4=hidden_b
    //   output: v3

    GraphRewriteSession session(graph);

    // Replace n0 with a single Embedding node using different inputs
    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "hidden_a")},
            .op_params = EmbeddingParams{},
            .debug_name = "replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).op_type, OpType::kEmbedding);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).inputs[0],
              committed->GetInputs()[1].value);// tokens_b
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphWithMultipleReplacements) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    // Graph layout:
    //   v0=tokens_a, v1=tokens_b, v2=weight
    //   n0=Embedding(v0,v2) → v3=hidden_a
    //   n1=Embedding(v1,v2) → v4=hidden_b
    //   output: v3

    GraphRewriteSession session(graph);

    // Replace n0 with two independent Embedding nodes
    ReplacementNode r1{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{.desc = HiddenDesc("unused_hidden")}},
            .op_params = EmbeddingParams{},
            .debug_name = "r1",
    };
    ReplacementNode r2{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "hidden_a")},
            .op_params = EmbeddingParams{},
            .debug_name = "r2",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(r1), std::move(r2)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 3U);// r1 + r2 + n1
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).name, "r1");
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 1}).name, "r2");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsReplacementOfExternalOutput) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 2}, "weight_hijack")},
            .op_params = EmbeddingParams{},
            .debug_name = "invalid_external_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};

    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsReplacementOfNonOldNodeOutput) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 4}, "future_hijack")},
            .op_params = EmbeddingParams{},
            .debug_name = "invalid_future_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};

    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsDuplicateRealReplacementTarget) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode first{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "first_hidden")},
            .op_params = EmbeddingParams{},
            .debug_name = "first_replacement",
    };
    ReplacementNode second{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "second_hidden")},
            .op_params = EmbeddingParams{},
            .debug_name = "second_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};

    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(first), std::move(second)});

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphWithEmptyReplacementsActsAsRemove) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const std::array old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, RemoveNodeOverridesPreviousReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 4}, "hidden_b")},
            .op_params = EmbeddingParams{},
            .debug_name = "replacement_should_be_cleared",
    };
    constexpr std::array old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).name, "");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphOverridesPreviousRemove) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 4}, "hidden_b")},
            .op_params = EmbeddingParams{},
            .debug_name = "replacement_after_remove",
    };
    const std::array old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 1}).name, "replacement_after_remove");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphWithExplicitOutputDesc) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "explicit_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "explicit_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetOutputs().size(), 1U);
    EXPECT_EQ(committed->GetValue(committed->GetOutputs()[0].value).name, "explicit_hidden");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphReplacesMultipleOldNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "subgraph_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "subgraph_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).name, "subgraph_replacement");
    ASSERT_EQ(committed->GetOutputs().size(), 1U);
    EXPECT_EQ(committed->GetValue(committed->GetOutputs()[0].value).name, "subgraph_hidden");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, RemoveNodeClearsOverlappingSubgraphRewrite) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "subgraph_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "subgraph_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).op_type, OpType::kEmbedding);
    ASSERT_EQ(committed->GetOutputs().size(), 1U);
    EXPECT_EQ(committed->GetValue(committed->GetOutputs()[0].value).name, "hidden_a");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ApplySupportsReplaceSubgraphMutation) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    SubgraphReplacement replace{
            .old_nodes = {GraphNodeId{.index = 0}, GraphNodeId{.index = 1}},
            .replacement_nodes = {ReplacementNode{
                    .op_type = OpType::kEmbedding,
                    .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
                    .outputs = {RewriteOutputBinding{
                            .desc = NodeOutputDesc{.payload = ActivationValue{},
                                                   .name = "applied_subgraph_hidden"},
                            .replaces = GraphValueId{.index = 3},
                    }},
                    .op_params = EmbeddingParams{},
                    .debug_name = "applied_subgraph_replacement",
            }},
    };
    const std::array<GraphMutation, 1> mutations{std::move(replace)};
    ASSERT_TRUE(session.Apply(std::span<const GraphMutation>{mutations}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).name, "applied_subgraph_replacement");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphSupportsVirtualInternalEdge) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    ReplacementNode first{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "virtual_hidden"},
                    .replaces = mid,
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "virtual_producer",
    };
    ReplacementNode second{
            .op_type = OpType::kAdd,
            .inputs = {mid, mid},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "final_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = AddParams{},
            .debug_name = "virtual_consumer",
    };
    constexpr std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(first), std::move(second)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 3U);
    const GraphNode& producer = committed->GetNode(GraphNodeId{.index = 0});
    const GraphNode& consumer = committed->GetNode(GraphNodeId{.index = 1});
    ASSERT_EQ(producer.outputs.size(), 1U);
    ASSERT_EQ(consumer.inputs.size(), 2U);
    EXPECT_EQ(consumer.inputs[0], producer.outputs[0]);
    EXPECT_EQ(consumer.inputs[1], producer.outputs[0]);
    EXPECT_EQ(committed->GetValue(committed->GetOutputs()[0].value).name, "final_hidden");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsUndefinedVirtualValueInput) {
    // Phase B: ReplaceSubgraph must fail at apply time when a virtual value is
    // consumed but never produced by any replacement in the same group. The
    // scratch map has no inferred spec for `mid`, so ResolveValueSpec returns
    // NotFound before InferOperator is even called.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    ReplacementNode replacement{
            .op_type = OpType::kAdd,
            .inputs = {mid, mid},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "final_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = AddParams{},
            .debug_name = "undefined_virtual_consumer",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsDuplicateVirtualValueProducer) {
    // Phase B: ReplaceSubgraph must fail at apply time when two replacement
    // nodes in the same group both try to produce the same virtual value.
    // ValidateReplacementSemantics detects the duplicate via the scratch map.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    ReplacementNode first{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "first_virtual"},
                    .replaces = mid,
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "first_virtual_producer",
    };
    ReplacementNode second{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "second_virtual"},
                    .replaces = mid,
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "second_virtual_producer",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(first), std::move(second)});
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsVirtualValueConsumedBeforeProducer) {
    // Phase B: ReplaceSubgraph must fail at apply time when a virtual value is
    // consumed by an earlier replacement than its producer. The consumer is
    // replayed first; ResolveValueSpec finds no inferred spec for `mid` and
    // returns NotFound.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    ReplacementNode consumer{
            .op_type = OpType::kAdd,
            .inputs = {mid, mid},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "final_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = AddParams{},
            .debug_name = "early_virtual_consumer",
    };
    ReplacementNode producer{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "virtual_hidden"},
                    .replaces = mid,
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "late_virtual_producer",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(consumer), std::move(producer)});
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsVirtualValueAcrossRewriteEntries) {
    // Phase B: each ReplaceSubgraph call validates its own replacement group
    // with a fresh scratch map. A virtual value produced in rewrite 1 is not
    // visible in rewrite 2's scratch map, so the consumer's ReplaceSubgraph
    // fails at apply time (ResolveValueSpec returns NotFound for `mid`).
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    ReplacementNode producer{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "virtual_hidden"},
                    .replaces = mid,
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "virtual_producer",
    };
    ReplacementNode consumer{
            .op_type = OpType::kAdd,
            .inputs = {mid, mid},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "final_hidden"},
                    .replaces = GraphValueId{.index = 4},
            }},
            .op_params = AddParams{},
            .debug_name = "cross_rewrite_consumer",
    };
    const std::array producer_old_nodes{GraphNodeId{.index = 0}};
    const std::array consumer_old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(producer_old_nodes, {std::move(producer)}).ok());
    const Status consumer_status = session.ReplaceSubgraph(consumer_old_nodes, {std::move(consumer)});
    ASSERT_FALSE(consumer_status.ok()) << consumer_status.ToString();
    EXPECT_EQ(consumer_status.code(), StatusCode::kNotFound);
}

TEST(GraphRewriteSession, ReplaceValueRejectsVirtualValues) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    const Status replace_old = session.ReplaceValue(mid, GraphValueId{.index = 0});
    const Status replace_new = session.ReplaceValue(GraphValueId{.index = 0}, mid);

    EXPECT_EQ(replace_old.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(replace_new.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsInvalidMapping) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "explicit_hidden"},
                    .replaces = GraphValueId{.index = 999},
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "invalid_mapping",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsInvalidInputId) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 999}},// invalid
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "hidden_a")},
            .op_params = EmbeddingParams{},
            .debug_name = "invalid_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
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
            NodeRemoval{.node = GraphNodeId{.index = 1}},
            InputRedirection{.node = GraphNodeId{.index = 0},
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
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    const GraphValueId k_cache = graph.AddState(
            Spec(DataType::Float32(), {2, 4, 8}),
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
            "kv_cache.layer_0.k");
    (void) k_cache;
    auto embed_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden"}},
            EmbeddingParams{});
    AM_CHECK(embed_or.ok(), "BuildGraphWithState embed AddNode failed");
    const AddedNode& embed = *embed_or;
    graph.MarkOutput(embed.outputs[0]);
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
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    auto embed_or = graph.AddNode(
            OpType::kEmbedding,
            3U,// decoder_layer_index = 3
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden"}},
            EmbeddingParams{});
    AM_CHECK(embed_or.ok(), "BuildGraphWithState embed AddNode failed");
    const AddedNode& embed = *embed_or;
    graph.MarkOutput(embed.outputs[0]);

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
                                        .name = "invalid"},
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
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "embed.weight");
    auto inline_data = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x01}, std::byte{0x02}});
    const GraphValueId constant = graph.AddConstant(
            Spec(DataType::Float32(), {1}),
            ConstantBinding{.inline_data = std::move(inline_data), .name = "scalar.one"},
            "one");
    auto embed_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden"}},
            EmbeddingParams{});
    AM_CHECK(embed_or.ok(), "BuildGraphWithState embed AddNode failed");
    const AddedNode& embed = *embed_or;
    graph.MarkOutput(embed.outputs[0]);
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
            EXPECT_EQ(value.name, "one");
        }
    }
    EXPECT_TRUE(found_constant);
}

TEST(GraphRewriteSession, AddConstantCommitsSessionConstantValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    auto inline_data = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x03}, std::byte{0x04}});
    const QuantizationSpec quantization{.kind = QuantizationKind::kInt8,
                                        .group_size = 64,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = false};

    const GraphValueId constant = session.AddConstant(
            Spec(DataType::Float32(), {1}),
            ConstantBinding{.inline_data = std::move(inline_data), .name = "folded.scalar"},
            quantization,
            "folded");
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());
    ASSERT_EQ(constant.index, graph.GetValues().size());

    bool found_constant = false;
    for (const GraphValue& value: committed->GetValues()) {
        if (const auto* c = std::get_if<ConstantValue>(&value.payload);
            c != nullptr && c->binding.name == "folded.scalar") {
            found_constant = true;
            ASSERT_TRUE(c->binding.inline_data != nullptr);
            EXPECT_EQ(c->binding.inline_data->size(), 2U);
            EXPECT_EQ(value.name, "folded");
            EXPECT_EQ(value.quantization, quantization);
            EXPECT_FALSE(value.producer.has_value());
        }
    }
    EXPECT_TRUE(found_constant);
}

TEST(GraphRewriteSession, AddConstantSupportsValueQueries) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const QuantizationSpec quantization{.kind = QuantizationKind::kInt4,
                                        .group_size = 32,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = true};

    const GraphValueId constant = session.AddConstant(
            Spec(DataType::Float32(), {2}),
            ConstantBinding{.name = "folded.vector"},
            quantization,
            "folded_vector");
    const GraphValueId virtual_value = session.AllocateVirtualValue();

    EXPECT_EQ(virtual_value.index, constant.index + 1U);
    EXPECT_TRUE(session.IsValueLive(constant));
    EXPECT_FALSE(session.IsValueLive(virtual_value));
    EXPECT_FALSE(session.IsGraphOutput(constant));
    EXPECT_TRUE(session.CheckValueId(constant).ok());
    EXPECT_EQ(session.CheckValueId(virtual_value).code(), StatusCode::kInvalidArgument);

    const StatusOr<GraphValueDesc> desc = session.GetValueOutputMetadata(constant);
    ASSERT_TRUE(desc.ok()) << desc.status().ToString();
    EXPECT_EQ(desc->spec, Spec(DataType::Float32(), {2}));
    const auto* constant_payload = std::get_if<ConstantValue>(&desc->payload);
    ASSERT_NE(constant_payload, nullptr);
    EXPECT_EQ(constant_payload->binding.name, "folded.vector");
    EXPECT_EQ(desc->quantization, quantization);
    EXPECT_EQ(desc->name, "folded_vector");

    const std::vector<GraphValueId> live = session.GetLiveValues();
    EXPECT_NE(std::ranges::find(live, constant), live.end());
    EXPECT_EQ(std::ranges::find(live, virtual_value), live.end());
}

TEST(GraphRewriteSession, ReplaceValueCanResolveToSessionConstant) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId constant = session.AddConstant(
            Spec(DataType::Float32(), {1, 1, 4}),
            ConstantBinding{.name = "folded.hidden"},
            {},
            "folded_hidden");

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 4}, constant).ok());
    EXPECT_EQ(session.GetResolvedValue(GraphValueId{.index = 4}), constant);
    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());
    bool found_constant = false;
    for (const GraphValue& value: committed->GetValues()) {
        if (const auto* constant_payload = std::get_if<ConstantValue>(&value.payload);
            constant_payload != nullptr && constant_payload->binding.name == "folded.hidden") {
            found_constant = true;
            EXPECT_EQ(value.name, "folded_hidden");
            EXPECT_EQ(value.quantization, QuantizationSpec{});
        }
    }
    EXPECT_TRUE(found_constant);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsSessionConstantReplacementTarget) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId constant = session.AddConstant(
            Spec(DataType::Float32(), {1}),
            ConstantBinding{.name = "session.constant"},
            {},
            "session_constant");
    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{.desc = HiddenDesc("replacement"),
                                             .replaces = constant}},
            .op_params = EmbeddingParams{},
            .debug_name = "replacement",
    };

    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
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

TEST(GraphRewriteSession, GetValueOutputMetadataReturnsFullValueDescriptor) {
    ModelGraph graph = BuildTwoEmbeddingGraph();
    const QuantizationSpec quantization{.kind = QuantizationKind::kInt8,
                                        .group_size = 64,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = false};
    graph.SetQuantization(GraphValueId{.index = 3}, quantization);

    GraphRewriteSession session(graph);
    const StatusOr<GraphValueDesc> desc = session.GetValueOutputMetadata(GraphValueId{.index = 3});

    ASSERT_TRUE(desc.ok()) << desc.status().ToString();
    EXPECT_EQ(desc->spec, graph.GetValue(GraphValueId{.index = 3}).spec);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(desc->payload));
    EXPECT_EQ(desc->quantization, quantization);
    EXPECT_EQ(desc->name, "hidden_a");
}

TEST(GraphRewriteSession, GetValueOutputMetadataRejectsInvalidValues) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId virtual_value = session.AllocateVirtualValue();

    EXPECT_EQ(session.GetValueOutputMetadata(virtual_value).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(session.GetValueOutputMetadata(GraphValueId{.index = 999}).status().code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, IsGraphOutputReportsDirectGraphOutputsOnly) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    EXPECT_TRUE(session.IsGraphOutput(GraphValueId{.index = 3}));
    EXPECT_FALSE(session.IsGraphOutput(GraphValueId{.index = 4}));

    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 3}, GraphValueId{.index = 4}).ok());
    EXPECT_TRUE(session.IsGraphOutput(GraphValueId{.index = 3}));
    EXPECT_FALSE(session.IsGraphOutput(GraphValueId{.index = 4}));

    EXPECT_FALSE(session.IsGraphOutput(session.AllocateVirtualValue()));
    EXPECT_FALSE(session.IsGraphOutput(GraphValueId{.index = 999}));
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

TEST(GraphRewriteSession, RedirectInputAccumulatesMultipleInputChanges) {
    // Build a graph with two weights so both inputs of an Embedding node
    // can be independently redirected to valid payload types.
    ModelGraph graph;
    const GraphValueId tokens_a = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens_a");
    const GraphValueId tokens_b = graph.AddInput(Spec(DataType::Int(32), {1}), "tokens_b");
    const GraphValueId weight_a = graph.AddWeight(Spec(DataType::Float32(), {16, 4}),
                                                  WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                                .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                  "embed_a.weight");
    const GraphValueId weight_b = graph.AddWeight(Spec(DataType::Float32(), {8, 4}),
                                                  WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                                .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                  "embed_b.weight");
    auto node_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens_a, weight_a},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden"}},
            EmbeddingParams{});
    ASSERT_TRUE(node_or.ok()) << node_or.status().ToString();
    const AddedNode& node = *node_or;
    graph.MarkOutput(node.outputs[0]);

    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, tokens_b).ok());
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 1, weight_b).ok());

    const StatusOr<GraphNodeView> view = session.GetNodeView(GraphNodeId{.index = 0});
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    ASSERT_EQ(view->inputs.size(), 2U);
    EXPECT_EQ(view->inputs[0], tokens_b);
    EXPECT_EQ(view->inputs[1], weight_b);

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    const GraphNode& committed_node = committed->GetNode(GraphNodeId{.index = 0});
    ASSERT_EQ(committed_node.inputs.size(), 2U);
    EXPECT_TRUE(std::holds_alternative<ModelInputValue>(
            committed->GetValue(committed_node.inputs[0]).payload));
    EXPECT_TRUE(std::holds_alternative<WeightValue>(
            committed->GetValue(committed_node.inputs[1]).payload));
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, RedirectInputRejectsExistingNodeReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "hidden_a")},
            .op_params = EmbeddingParams{},
            .debug_name = "arbitrary_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());
    ASSERT_FALSE(session.GetNodeView(GraphNodeId{.index = 0}).ok());

    const Status status = session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    ASSERT_FALSE(session.GetNodeView(GraphNodeId{.index = 0}).ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    const GraphNode& committed_node = committed->GetNode(GraphNodeId{.index = 0});
    EXPECT_EQ(committed_node.name, "arbitrary_replacement");
    EXPECT_EQ(committed_node.inputs[0], committed->GetInputs()[0].value);
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(GraphRewriteSession, RedirectInputRejectsOverlappingSubgraphRewrite) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "subgraph_hidden"},
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = EmbeddingParams{},
            .debug_name = "subgraph_replacement",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());
    const Status status = session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_EQ(committed->GetNode(GraphNodeId{.index = 0}).name, "subgraph_replacement");
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(SubgraphBuilder, EmitsAndYieldsReplacementSubgraph) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}});

    auto mid_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            HiddenDesc("embed_mid_output"),
            EmbeddingParams{},
            std::nullopt,
            "embed_mid");
    ASSERT_TRUE(mid_or.ok()) << mid_or.status().ToString();
    const GraphValueId mid = std::move(*mid_or);
    ASSERT_NE(mid.index, 0U);

    ASSERT_TRUE(builder.Yield(mid, GraphValueId{.index = 3}).ok());

    ASSERT_TRUE(builder.Commit().ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 2U);

    const GraphNode& new_embed = committed->GetNode(GraphNodeId{.index = 0});
    EXPECT_EQ(new_embed.op_type, OpType::kEmbedding);
    ASSERT_EQ(new_embed.outputs.size(), 1U);
    EXPECT_EQ(new_embed.inputs[0], committed->GetInputs()[1].value);
    EXPECT_EQ(committed->GetOutputs()[0].value, new_embed.outputs[0]);
    EXPECT_TRUE(committed->Validate().ok());
}

TEST(SubgraphBuilder, EmitAcceptsFullOutputDesc) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const QuantizationSpec quantization{.kind = QuantizationKind::kInt8,
                                        .group_size = 64,
                                        .scale_dtype = DataType::Float32(),
                                        .has_zero_point = false};
    NodeOutputDesc output_desc{.payload = ActivationValue{},
                               .quantization = quantization,
                               .name = "builder_full_desc"};

    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}});
    auto out_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            output_desc,
            EmbeddingParams{},
            std::nullopt,
            "embed_full_desc");
    ASSERT_TRUE(out_or.ok()) << out_or.status().ToString();
    const GraphValueId out = std::move(*out_or);
    ASSERT_TRUE(builder.Yield(out, GraphValueId{.index = 3}).ok());
    ASSERT_TRUE(builder.Commit().ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());
    ASSERT_EQ(committed->GetOutputs().size(), 1U);

    const GraphValue& output = committed->GetValue(committed->GetOutputs()[0].value);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(output.payload));
    EXPECT_EQ(output.quantization, quantization);
    EXPECT_EQ(output.name, "builder_full_desc");
}

TEST(SubgraphBuilder, EmitReturnsVirtualValueForEachOutputDesc) {
    const ModelGraph graph = BuildRoPEGraph();
    GraphRewriteSession session(graph);
    const GraphNode& rope = graph.GetNode(GraphNodeId{.index = 2});

    SubgraphBuilder builder(session, {GraphNodeId{.index = 2}});
    auto outputs_or = builder.Emit(
            OpType::kRoPE,
            rope.inputs,
            std::vector<NodeOutputDesc>{HiddenDesc("q_rope_rewritten"), HiddenDesc("k_rope_rewritten")},
            ValidRoPEParams(),
            0U,
            "rope_rewritten");
    ASSERT_TRUE(outputs_or.ok()) << outputs_or.status().ToString();
    const std::vector<GraphValueId> outputs = std::move(*outputs_or);

    ASSERT_EQ(outputs.size(), 2U);
    EXPECT_NE(outputs[0], outputs[1]);
    EXPECT_GE(outputs[0].index, graph.GetValues().size());
    EXPECT_GE(outputs[1].index, graph.GetValues().size());
}

TEST(SubgraphBuilder, EmitMultiOutputYieldsEachOutputToDistinctTarget) {
    const ModelGraph graph = BuildRoPEGraph();
    GraphRewriteSession session(graph);
    const GraphNode& rope = graph.GetNode(GraphNodeId{.index = 2});
    const QuantizationSpec q_quantization{.kind = QuantizationKind::kInt8,
                                          .group_size = 64,
                                          .scale_dtype = DataType::Float32(),
                                          .has_zero_point = false};
    const QuantizationSpec k_quantization{.kind = QuantizationKind::kInt4,
                                          .group_size = 32,
                                          .scale_dtype = DataType::Float32(),
                                          .has_zero_point = true};
    NodeOutputDesc q_desc = HiddenDesc("q_rope_rewritten");
    q_desc.quantization = q_quantization;
    NodeOutputDesc k_desc = HiddenDesc("k_rope_rewritten");
    k_desc.quantization = k_quantization;

    SubgraphBuilder builder(session, {GraphNodeId{.index = 2}});
    auto outputs_or = builder.Emit(
            OpType::kRoPE,
            rope.inputs,
            std::vector<NodeOutputDesc>{q_desc, k_desc},
            ValidRoPEParams(),
            0U,
            "rope_rewritten");
    ASSERT_TRUE(outputs_or.ok()) << outputs_or.status().ToString();
    const std::vector<GraphValueId> outputs = std::move(*outputs_or);
    ASSERT_EQ(outputs.size(), 2U);
    ASSERT_TRUE(builder.Yield(outputs[0], rope.outputs[0]).ok());
    ASSERT_TRUE(builder.Yield(outputs[1], rope.outputs[1]).ok());
    ASSERT_TRUE(builder.Commit().ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());

    const GraphNode& committed_rope = committed->GetNode(GraphNodeId{.index = 2});
    EXPECT_EQ(committed_rope.op_type, OpType::kRoPE);
    ASSERT_EQ(committed_rope.outputs.size(), 2U);
    EXPECT_EQ(committed->GetOutputs()[0].value, committed_rope.outputs[0]);
    EXPECT_EQ(committed->GetOutputs()[1].value, committed_rope.outputs[1]);
    EXPECT_EQ(committed->GetValue(committed_rope.outputs[0]).name, "q_rope_rewritten");
    EXPECT_EQ(committed->GetValue(committed_rope.outputs[0]).quantization, q_quantization);
    EXPECT_EQ(committed->GetValue(committed_rope.outputs[1]).name, "k_rope_rewritten");
    EXPECT_EQ(committed->GetValue(committed_rope.outputs[1]).quantization, k_quantization);
}

TEST(SubgraphBuilder, EmitMultiOutputFeedsSubsequentEmit) {
    const ModelGraph graph = BuildRoPEGraph();
    GraphRewriteSession session(graph);
    const GraphNode& rope = graph.GetNode(GraphNodeId{.index = 2});

    SubgraphBuilder builder(session, {GraphNodeId{.index = 2}});
    auto rope_outputs_or = builder.Emit(
            OpType::kRoPE,
            rope.inputs,
            std::vector<NodeOutputDesc>{HiddenDesc("q_rope_internal"), HiddenDesc("k_rope_forwarded")},
            ValidRoPEParams(),
            0U,
            "rope_rewritten");
    ASSERT_TRUE(rope_outputs_or.ok()) << rope_outputs_or.status().ToString();
    const std::vector<GraphValueId> rope_outputs = std::move(*rope_outputs_or);
    ASSERT_EQ(rope_outputs.size(), 2U);
    auto sum_or = builder.Emit(
            OpType::kAdd,
            {rope_outputs[0], rope_outputs[0]},
            HiddenDesc("q_rope_summed"),
            AddParams{});
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = std::move(*sum_or);
    ASSERT_TRUE(builder.Yield(sum, rope.outputs[0]).ok());
    ASSERT_TRUE(builder.Yield(rope_outputs[1], rope.outputs[1]).ok());
    ASSERT_TRUE(builder.Commit().ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_TRUE(committed->Validate().ok());

    ASSERT_EQ(committed->GetNodes().size(), 4U);
    const GraphNode& committed_rope = committed->GetNode(GraphNodeId{.index = 2});
    const GraphNode& committed_add = committed->GetNode(GraphNodeId{.index = 3});
    ASSERT_EQ(committed_rope.outputs.size(), 2U);
    ASSERT_EQ(committed_add.inputs.size(), 2U);
    EXPECT_EQ(committed_add.inputs[0], committed_rope.outputs[0]);
    EXPECT_EQ(committed_add.inputs[1], committed_rope.outputs[0]);
    EXPECT_EQ(committed->GetOutputs()[0].value, committed_add.outputs[0]);
    EXPECT_EQ(committed->GetOutputs()[1].value, committed_rope.outputs[1]);
}

TEST(SubgraphBuilder, YieldRejectsUnknownInternalValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}});
    auto mid_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            HiddenDesc("unknown_internal_output"),
            EmbeddingParams{});
    ASSERT_TRUE(mid_or.ok()) << mid_or.status().ToString();
    const GraphValueId mid = std::move(*mid_or);

    const GraphValueId unproduced = session.AllocateVirtualValue();
    const Status status = builder.Yield(unproduced, GraphValueId{.index = 3});
    EXPECT_FALSE(status.ok());

    (void) mid;
}

TEST(SubgraphBuilder, YieldRejectsVirtualValueAsReplacementTarget) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}});
    auto mid_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            HiddenDesc("virtual_target_output"),
            EmbeddingParams{});
    ASSERT_TRUE(mid_or.ok()) << mid_or.status().ToString();
    const GraphValueId mid = std::move(*mid_or);

    const GraphValueId another_virtual = session.AllocateVirtualValue();
    const Status status = builder.Yield(mid, another_virtual);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SubgraphBuilder, YieldRejectsOutOfRangeReplacementTarget) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}});
    auto mid_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            HiddenDesc("out_of_range_target_output"),
            EmbeddingParams{});
    ASSERT_TRUE(mid_or.ok()) << mid_or.status().ToString();
    const GraphValueId mid = std::move(*mid_or);

    const GraphValueId out_of_range{.index = 999};
    const Status status = builder.Yield(mid, out_of_range);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SubgraphBuilder, BuilderReusableAfterCommit) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}});
    auto out1_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            HiddenDesc("first_reusable_output"),
            EmbeddingParams{});
    ASSERT_TRUE(out1_or.ok()) << out1_or.status().ToString();
    const GraphValueId out1 = std::move(*out1_or);
    ASSERT_TRUE(builder.Yield(out1, GraphValueId{.index = 3}).ok());
    ASSERT_TRUE(builder.Commit().ok());

    auto out2_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            HiddenDesc("second_reusable_output"),
            EmbeddingParams{});
    ASSERT_TRUE(out2_or.ok()) << out2_or.status().ToString();
    const GraphValueId out2 = std::move(*out2_or);
    ASSERT_TRUE(builder.Yield(out2, GraphValueId{.index = 3}).ok());
    ASSERT_TRUE(builder.Commit().ok());

    const StatusOr<ModelGraph> committed = session.Commit();
    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    ASSERT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_TRUE(committed->Validate().ok());
}

// --- P0: Node enumeration API (IsNodeLive / GetTopologicalOrder /
//         FindNodesByOpType). Passes must be able to discover nodes
//         independently instead of relying on externally-supplied ids. ---

// Graph layout for all tests below:
//   v0=tokens_a, v1=tokens_b, v2=weight
//   n0=Embedding(v0,v2) -> v3=hidden_a (graph output)
//   n1=Embedding(v1,v2) -> v4=hidden_b

ReplacementNode MakeReplacementEmbedding(GraphValueId replaces, const char* debug_name) {
    return ReplacementNode{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(replaces, debug_name)},
            .op_params = EmbeddingParams{},
            .debug_name = debug_name,
    };
}

TEST(GraphRewriteSession, IsNodeLiveReturnsTrueForAllNodesOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    EXPECT_TRUE(session.IsNodeLive(GraphNodeId{.index = 0}));
    EXPECT_TRUE(session.IsNodeLive(GraphNodeId{.index = 1}));
}

TEST(GraphRewriteSession, IsNodeLiveReturnsFalseForRemovedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());

    EXPECT_TRUE(session.IsNodeLive(GraphNodeId{.index = 0}));
    EXPECT_FALSE(session.IsNodeLive(GraphNodeId{.index = 1}));
}

TEST(GraphRewriteSession, IsNodeLiveReturnsTrueForRedirectedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());

    // RedirectInput installs a mirror replacement that still exposes the
    // original node identity, so the node remains live.
    EXPECT_TRUE(session.IsNodeLive(GraphNodeId{.index = 0}));
    EXPECT_TRUE(session.IsNodeLive(GraphNodeId{.index = 1}));
}

TEST(GraphRewriteSession, IsNodeLiveReturnsFalseForReplacedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 3},
                                                                              "replacement")})
                        .ok());

    EXPECT_FALSE(session.IsNodeLive(GraphNodeId{.index = 0}));
    EXPECT_TRUE(session.IsNodeLive(GraphNodeId{.index = 1}));
}

TEST(GraphRewriteSession, IsNodeLiveStaysFalseAfterRejectedRedirectOnRemovedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());
    EXPECT_FALSE(session.IsNodeLive(GraphNodeId{.index = 1}));

    const Status status = session.RedirectInput(GraphNodeId{.index = 1}, 0, GraphValueId{.index = 0});
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(session.IsNodeLive(GraphNodeId{.index = 1}));
}

TEST(GraphRewriteSession, IsNodeLiveReturnsFalseForOutOfRangeId) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    EXPECT_FALSE(session.IsNodeLive(GraphNodeId{.index = 999}));
}

TEST(GraphRewriteSession, GetTopologicalOrderReturnsAllNodesOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const StatusOr<std::vector<GraphNodeId>> expected = graph.TopologicalOrder();
    ASSERT_TRUE(expected.ok()) << expected.status().ToString();

    const StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), expected->size());
    for (size_t i = 0; i < order->size(); ++i) {
        EXPECT_EQ((*order)[i], (*expected)[i]) << "Mismatch at index " << i;
    }
}

TEST(GraphRewriteSession, GetTopologicalOrderExcludesRemovedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());

    const StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), 1U);
    EXPECT_EQ((*order)[0], GraphNodeId{.index = 0});
}

TEST(GraphRewriteSession, GetTopologicalOrderIncludesRedirectedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());

    const StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), 2U);
}

TEST(GraphRewriteSession, GetTopologicalOrderExcludesReplacedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 3},
                                                                              "replacement")})
                        .ok());

    const StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), 1U);
    EXPECT_EQ((*order)[0], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindNodesByOpTypeReturnsAllMatchesOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const std::vector<GraphNodeId> found = session.FindNodesByOpType(OpType::kEmbedding);
    ASSERT_EQ(found.size(), 2U);
    EXPECT_EQ(found[0], GraphNodeId{.index = 0});
    EXPECT_EQ(found[1], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindNodesByOpTypeExcludesRemovedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 0}).ok());

    const std::vector<GraphNodeId> found = session.FindNodesByOpType(OpType::kEmbedding);
    ASSERT_EQ(found.size(), 1U);
    EXPECT_EQ(found[0], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindNodesByOpTypeExcludesReplacedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const std::array old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 4},
                                                                              "replacement")})
                        .ok());

    const std::vector<GraphNodeId> found = session.FindNodesByOpType(OpType::kEmbedding);
    ASSERT_EQ(found.size(), 1U);
    EXPECT_EQ(found[0], GraphNodeId{.index = 0});
}

TEST(GraphRewriteSession, FindNodesByOpTypeReturnsEmptyForNonExistentOpType) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const std::vector<GraphNodeId> found = session.FindNodesByOpType(OpType::kSoftmax);
    EXPECT_TRUE(found.empty());
}

TEST(GraphRewriteSession, FindNodesByOpTypeIncludesRedirectedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());

    const std::vector<GraphNodeId> found = session.FindNodesByOpType(OpType::kEmbedding);
    ASSERT_EQ(found.size(), 2U);
    EXPECT_EQ(found[0], GraphNodeId{.index = 0});
    EXPECT_EQ(found[1], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, GetTopologicalOrderConsistentWithIsNodeLive) {
    // After a mix of mutations (remove n0, redirect n1 input, replace value),
    // every id in GetTopologicalOrder() must satisfy IsNodeLive==true, and
    // every live id must appear in GetTopologicalOrder().
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 0}).ok());
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 1}, 0, GraphValueId{.index = 0})
                        .ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 3}, GraphValueId{.index = 4}).ok());

    const StatusOr<std::vector<GraphNodeId>> order = session.GetTopologicalOrder();
    ASSERT_TRUE(order.ok()) << order.status().ToString();

    for (GraphNodeId id: *order) {
        EXPECT_TRUE(session.IsNodeLive(id))
                << "TopologicalOrder contains non-live node " << id.index;
    }

    for (uint32_t i = 0; i < graph.GetNodes().size(); ++i) {
        const GraphNodeId id{.index = i};
        if (session.IsNodeLive(id)) {
            EXPECT_NE(std::find(order->begin(), order->end(), id), order->end())
                    << "Live node " << i << " missing from TopologicalOrder";
        }
    }
}

TEST(GraphRewriteSession, IsValueLiveReturnsTrueForAllValuesOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // v0=tokens_a, v1=tokens_b, v2=weight (external); v3=hidden_a, v4=hidden_b (node-produced)
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 0}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 1}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 2}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 3}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 4}));
}

TEST(GraphRewriteSession, IsValueLiveReturnsFalseForOutOfRangeValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    EXPECT_FALSE(session.IsValueLive(GraphValueId{.index = 99}));
}

TEST(GraphRewriteSession, IsValueLiveReturnsFalseForVirtualValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId virtual_value = session.AllocateVirtualValue();
    EXPECT_FALSE(session.IsValueLive(virtual_value));
}

TEST(GraphRewriteSession, IsValueLiveExcludesValuesFromRemovedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // node 1 produces v4 (hidden_b)
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());
    // External values unaffected
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 0}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 1}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 2}));
    // node 0 still live → v3 live
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 3}));
    // node 1 removed → v4 dead
    EXPECT_FALSE(session.IsValueLive(GraphValueId{.index = 4}));
}

TEST(GraphRewriteSession, IsValueLiveIncludesValuesReplacedBySubgraph) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // node 0 produces v3 (hidden_a); replace it with a subgraph replacement
    // whose output `replaces` v3. v3 is still observable (graph output, mapped
    // to the replacement output at Commit time).
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 3},
                                                                              "replacement")})
                        .ok());
    // node 0 replaced but replacement takes over v3 → v3 live
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 3}));
    // node 1 still live → v4 live
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 4}));
    // External values unaffected
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 0}));
}

TEST(GraphRewriteSession, IsValueLiveExcludesValuesNotReplacedBySubgraph) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // Replace both n0 and n1, but the replacement only takes over v3.
    // v4 (produced by n1) has no replacement → dead.
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 3},
                                                                              "replacement")})
                        .ok());
    // v3 taken over by replacement → live
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 3}));
    // v4 not taken over, producer n1 replaced → dead
    EXPECT_FALSE(session.IsValueLive(GraphValueId{.index = 4}));
    // External values unaffected
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 0}));
}

TEST(GraphRewriteSession, IsValueLiveIncludesValuesFromRedirectedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // RedirectInput keeps the node live (mirror replacement), so its outputs stay live
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 3}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 4}));
}

TEST(GraphRewriteSession, IsValueLiveUnaffectedByReplaceValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // ReplaceValue redirects consumers but does not remove the value
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 3}, GraphValueId{.index = 4}).ok());
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 3}));
    EXPECT_TRUE(session.IsValueLive(GraphValueId{.index = 4}));
}

TEST(GraphRewriteSession, GetLiveValuesReturnsAllValuesOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const std::vector<GraphValueId> live = session.GetLiveValues();
    ASSERT_EQ(live.size(), 5U);
    EXPECT_EQ(live[0], GraphValueId{.index = 0});
    EXPECT_EQ(live[1], GraphValueId{.index = 1});
    EXPECT_EQ(live[2], GraphValueId{.index = 2});
    EXPECT_EQ(live[3], GraphValueId{.index = 3});
    EXPECT_EQ(live[4], GraphValueId{.index = 4});
}

TEST(GraphRewriteSession, GetLiveValuesExcludesValuesFromRemovedNode) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());
    const std::vector<GraphValueId> live = session.GetLiveValues();
    ASSERT_EQ(live.size(), 4U);
    // v4 (hidden_b, produced by removed node 1) excluded
    EXPECT_EQ(live[0], GraphValueId{.index = 0});
    EXPECT_EQ(live[1], GraphValueId{.index = 1});
    EXPECT_EQ(live[2], GraphValueId{.index = 2});
    EXPECT_EQ(live[3], GraphValueId{.index = 3});
}

TEST(GraphRewriteSession, GetLiveValuesIncludesValuesReplacedBySubgraph) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 3},
                                                                              "replacement")})
                        .ok());
    const std::vector<GraphValueId> live = session.GetLiveValues();
    // v3 is replaced by the replacement output → still live
    ASSERT_EQ(live.size(), 5U);
    EXPECT_EQ(live[0], GraphValueId{.index = 0});
    EXPECT_EQ(live[1], GraphValueId{.index = 1});
    EXPECT_EQ(live[2], GraphValueId{.index = 2});
    EXPECT_EQ(live[3], GraphValueId{.index = 3});
    EXPECT_EQ(live[4], GraphValueId{.index = 4});
}

TEST(GraphRewriteSession, GetLiveValuesExcludesValuesNotReplacedBySubgraph) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // Replace both n0 and n1, but the replacement only takes over v3.
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 3},
                                                                              "replacement")})
                        .ok());
    const std::vector<GraphValueId> live = session.GetLiveValues();
    // v4 (produced by replaced n1, no replacement takes over) excluded
    ASSERT_EQ(live.size(), 4U);
    EXPECT_EQ(live[0], GraphValueId{.index = 0});
    EXPECT_EQ(live[1], GraphValueId{.index = 1});
    EXPECT_EQ(live[2], GraphValueId{.index = 2});
    EXPECT_EQ(live[3], GraphValueId{.index = 3});
}

TEST(GraphRewriteSession, GetLiveValuesConsistentWithIsValueLive) {
    // After a mix of mutations (remove n0, redirect n1, replace value),
    // every id in GetLiveValues() must satisfy IsValueLive==true, and
    // every live id must appear in GetLiveValues().
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 0}).ok());
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 1}, 0, GraphValueId{.index = 0})
                        .ok());
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 3}, GraphValueId{.index = 4}).ok());

    const std::vector<GraphValueId> live = session.GetLiveValues();

    for (GraphValueId id: live) {
        EXPECT_TRUE(session.IsValueLive(id))
                << "GetLiveValues contains non-live value " << id.index;
    }

    for (uint32_t i = 0; i < graph.GetValues().size(); ++i) {
        const GraphValueId id{.index = i};
        if (session.IsValueLive(id)) {
            EXPECT_NE(std::ranges::find(live, id), live.end())
                    << "Live value " << i << " missing from GetLiveValues";
        }
    }
}

TEST(GraphRewriteSession, FindConsumersReturnsAllConsumersOnCleanSession) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // v2 (weight) is consumed by both n0 and n1
    const std::vector<GraphNodeId> consumers = session.FindConsumers(GraphValueId{.index = 2});
    ASSERT_EQ(consumers.size(), 2U);
    EXPECT_EQ(consumers[0], GraphNodeId{.index = 0});
    EXPECT_EQ(consumers[1], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindConsumersReturnsSingleConsumer) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // v0 (tokens_a) is consumed only by n0
    const std::vector<GraphNodeId> consumers = session.FindConsumers(GraphValueId{.index = 0});
    ASSERT_EQ(consumers.size(), 1U);
    EXPECT_EQ(consumers[0], GraphNodeId{.index = 0});
}

TEST(GraphRewriteSession, FindConsumersReturnsEmptyForValueWithNoConsumers) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // v3 (hidden_a) is a graph output, no node consumes it
    const std::vector<GraphNodeId> consumers = session.FindConsumers(GraphValueId{.index = 3});
    EXPECT_TRUE(consumers.empty());
}

TEST(GraphRewriteSession, FindConsumersExcludesRemovedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());
    // v2 (weight) now consumed only by n0 (n1 removed)
    const std::vector<GraphNodeId> consumers = session.FindConsumers(GraphValueId{.index = 2});
    ASSERT_EQ(consumers.size(), 1U);
    EXPECT_EQ(consumers[0], GraphNodeId{.index = 0});
}

TEST(GraphRewriteSession, FindConsumersExcludesReplacedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const std::array old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session
                        .ReplaceSubgraph(old_nodes, {MakeReplacementEmbedding(GraphValueId{.index = 4},
                                                                              "replacement")})
                        .ok());
    // v2 (weight) now consumed only by n0 (n1 replaced, not live)
    const std::vector<GraphNodeId> consumers = session.FindConsumers(GraphValueId{.index = 2});
    ASSERT_EQ(consumers.size(), 1U);
    EXPECT_EQ(consumers[0], GraphNodeId{.index = 0});
}

TEST(GraphRewriteSession, FindConsumersReflectsRedirectInput) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // Redirect n0's input 0 from v0 to v1
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());
    // n0 no longer consumes v0
    EXPECT_TRUE(session.FindConsumers(GraphValueId{.index = 0}).empty());
    // n0 now consumes v1 (along with n1)
    const std::vector<GraphNodeId> v1_consumers = session.FindConsumers(GraphValueId{.index = 1});
    ASSERT_EQ(v1_consumers.size(), 2U);
    EXPECT_EQ(v1_consumers[0], GraphNodeId{.index = 0});
    EXPECT_EQ(v1_consumers[1], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindConsumersCacheInvalidatesAfterRedirectInput) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_EQ(session.FindConsumers(GraphValueId{.index = 0}).size(), 1U);
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());

    EXPECT_TRUE(session.FindConsumers(GraphValueId{.index = 0}).empty());
    const std::vector<GraphNodeId> consumers = session.FindConsumers(GraphValueId{.index = 1});
    ASSERT_EQ(consumers.size(), 2U);
    EXPECT_EQ(consumers[0], GraphNodeId{.index = 0});
    EXPECT_EQ(consumers[1], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindConsumersResolvesReplaceValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // Replace v2 (weight) with v0 (tokens_a): consumers of v2 now resolve to v0
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 2}, GraphValueId{.index = 0}).ok());
    // FindConsumers(v0) should return both n0 and n1 (both originally consumed v2 → v0)
    const std::vector<GraphNodeId> v0_consumers = session.FindConsumers(GraphValueId{.index = 0});
    ASSERT_EQ(v0_consumers.size(), 2U);
    EXPECT_EQ(v0_consumers[0], GraphNodeId{.index = 0});
    EXPECT_EQ(v0_consumers[1], GraphNodeId{.index = 1});
    // FindConsumers(v2) should return the same result (v2 resolves to v0)
    const std::vector<GraphNodeId> v2_consumers = session.FindConsumers(GraphValueId{.index = 2});
    ASSERT_EQ(v2_consumers.size(), 2U);
    EXPECT_EQ(v2_consumers[0], GraphNodeId{.index = 0});
    EXPECT_EQ(v2_consumers[1], GraphNodeId{.index = 1});
}

TEST(GraphRewriteSession, FindConsumersReturnsEmptyForVirtualValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId virtual_value = session.AllocateVirtualValue();
    EXPECT_TRUE(session.FindConsumers(virtual_value).empty());
}

TEST(GraphRewriteSession, FindConsumersReturnsEmptyForOutOfRangeValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    EXPECT_TRUE(session.FindConsumers(GraphValueId{.index = 99}).empty());
}

TEST(GraphRewriteSession, HasLiveConsumersReturnsTrueForConsumedValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // v2 (weight) is consumed by both n0 and n1
    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 2}));
}

TEST(GraphRewriteSession, HasLiveConsumersReturnsFalseForValueWithNoConsumers) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // v3 (hidden_a) is a graph output, no node consumes it
    EXPECT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 3}));
}

TEST(GraphRewriteSession, HasLiveConsumersReturnsFalseForVirtualValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId virtual_value = session.AllocateVirtualValue();
    EXPECT_FALSE(session.HasLiveConsumers(virtual_value));
}

TEST(GraphRewriteSession, HasLiveConsumersReturnsFalseForOutOfRangeValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    EXPECT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 99}));
}

TEST(GraphRewriteSession, HasLiveConsumersExcludesRemovedNodes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());
    // v1 (tokens_b) was only consumed by n1; after removing n1, no live consumer
    EXPECT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 1}));
    // v2 (weight) still consumed by n0
    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 2}));
}

TEST(GraphRewriteSession, HasLiveConsumersReflectsRedirectInput) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // Redirect n0's input 0 from v0 to v1
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1})
                        .ok());
    // v0 (tokens_a) no longer consumed by any live node
    EXPECT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 0}));
    // v1 (tokens_b) now consumed by both n0 and n1
    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 1}));
}

TEST(GraphRewriteSession, HasLiveConsumersCacheInvalidatesAfterReplaceValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 3}));
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 2}, GraphValueId{.index = 3}).ok());

    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 3}));
    EXPECT_EQ(session.HasLiveConsumers(GraphValueId{.index = 2}),
              session.HasLiveConsumers(GraphValueId{.index = 3}));
}

TEST(GraphRewriteSession, HasLiveConsumersResolvesReplaceValue) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    // Replace v2 (weight) with v0 (tokens_a)
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 2}, GraphValueId{.index = 0}).ok());
    // HasLiveConsumers(v2) should match HasLiveConsumers(v0): both resolve to v0
    EXPECT_EQ(session.HasLiveConsumers(GraphValueId{.index = 2}),
              session.HasLiveConsumers(GraphValueId{.index = 0}));
    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 2}));
}

TEST(GraphRewriteSession, HasLiveConsumersTrueWhenOnlyReplacementConsumes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    // Graph layout:
    //   v0=tokens_a, v1=tokens_b, v2=weight
    //   n0=Embedding(v0,v2) → v3=hidden_a
    //   n1=Embedding(v1,v2) → v4=hidden_b
    GraphRewriteSession session(graph);

    // Replace both n0 and n1 with a single replacement that consumes v2.
    // After this, no live original node consumes v2, but the replacement does.
    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "merged")},
            .op_params = EmbeddingParams{},
            .debug_name = "merged",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    // FindConsumers returns empty: no live original node consumes v2.
    EXPECT_TRUE(session.FindConsumers(GraphValueId{.index = 2}).empty());
    // HasLiveConsumers returns true: the replacement node consumes v2.
    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 2}));
}

TEST(GraphRewriteSession, HasLiveConsumersFalseWhenReplacementDoesNotConsume) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    // alt_weight is a session constant of Float32[16,4] (same spec as v2) so
    // the merged Embedding has a valid dtype chain without consuming v2.
    const GraphValueId alt_weight = session.AddConstant(
            Spec(DataType::Float32(), {16, 4}),
            ConstantBinding{.name = "alt.weight"},
            QuantizationSpec{},
            "alt.weight");

    // Replace both n0 and n1 with a single replacement that does NOT consume v2.
    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, alt_weight},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "merged")},
            .op_params = EmbeddingParams{},
            .debug_name = "merged",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    // No live original node and no replacement consumes v2.
    EXPECT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 2}));
}

TEST(GraphRewriteSession, HasLiveConsumersCacheInvalidatesAfterReplaceSubgraph) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ASSERT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 2}));

    // alt_weight supplies a valid Float32 weight slot without consuming v2.
    const GraphValueId alt_weight = session.AddConstant(
            Spec(DataType::Float32(), {16, 4}),
            ConstantBinding{.name = "alt.weight"},
            QuantizationSpec{},
            "alt.weight");

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, alt_weight},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "merged")},
            .op_params = EmbeddingParams{},
            .debug_name = "merged",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}, GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    EXPECT_FALSE(session.HasLiveConsumers(GraphValueId{.index = 2}));
}

TEST(GraphRewriteSession, HasLiveConsumersTrueForUpstreamValueConsumedByReplacement) {
    // Scenario from the bug report: a live value's only original consumer was
    // replaced, but the replacement node still consumes the value. DCE must not
    // delete the value's producer.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    // Replace n0 (which consumed v0) with r that also consumes v0.
    // Remove n1 so no other live node consumes v2 (isolate the case).
    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "r")},
            .op_params = EmbeddingParams{},
            .debug_name = "r",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());
    ASSERT_TRUE(session.RemoveNode(GraphNodeId{.index = 1}).ok());

    // v0 (tokens_a): FindConsumers empty (n0 replaced, n1 never consumed v0),
    // but HasLiveConsumers true (replacement r consumes v0).
    EXPECT_TRUE(session.FindConsumers(GraphValueId{.index = 0}).empty());
    EXPECT_TRUE(session.HasLiveConsumers(GraphValueId{.index = 0}));
}


TEST(SubgraphBuilder, CommitRejectsWrongDtypeChain) {
    // Phase B: builder.Commit() must fail when an Emit's output dtype is
    // incompatible with the slot dtype of a subsequent Emit's input. Here the
    // first Emit produces a Float32 hidden (Embedding output), and the second
    // Emit feeds that Float32 into the Int32 tokens slot of another Embedding.
    // InferOperator rejects this when ReplaceSubgraph replays the chain.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    SubgraphBuilder builder(session, {GraphNodeId{.index = 0}, GraphNodeId{.index = 1}});

    auto first_or = builder.Emit(
            OpType::kEmbedding,
            {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            HiddenDesc("first_hidden"),
            EmbeddingParams{},
            std::nullopt,
            "first_embed");
    ASSERT_TRUE(first_or.ok()) << first_or.status().ToString();
    const GraphValueId first_hidden = *first_or;

    auto second_or = builder.Emit(
            OpType::kEmbedding,
            // first_hidden is Float32 (Embedding output) but the tokens slot
            // requires Int32/Int64/UInt32 - dtype chain violation.
            {first_hidden, GraphValueId{.index = 2}},
            HiddenDesc("second_hidden"),
            EmbeddingParams{},
            std::nullopt,
            "second_embed");
    // Emit itself does not validate dtype chains; it just accumulates the node.
    ASSERT_TRUE(second_or.ok()) << second_or.status().ToString();
    const GraphValueId second_hidden = *second_or;

    ASSERT_TRUE(builder.Yield(second_hidden, GraphValueId{.index = 3}).ok());

    const Status commit_status = builder.Commit();
    ASSERT_FALSE(commit_status.ok()) << commit_status.ToString();
    EXPECT_EQ(commit_status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsYieldSpecMismatch) {
    // Phase B: ReplaceSubgraph must fail when a replacement's inferred output
    // dtype does not match the dtype of the value it replaces (yield target).
    // Here an Argmax produces an Int64 index, but its binding tries to replace
    // v3 (Float32 hidden_a) - dtype mismatch.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    ReplacementNode replacement{
            .op_type = OpType::kArgmax,
            // v4 (Float32 hidden_b) is a valid Argmax input. n1 is not in
            // old_nodes, so v4 is still produced by a live original node.
            .inputs = {GraphValueId{.index = 4}},
            .outputs = {RewriteOutputBinding{
                    .desc = NodeOutputDesc{.payload = ActivationValue{},
                                           .name = "argmax_output"},
                    // v3 (Float32 hidden_a) is the replaces target; Argmax
                    // infers Int64 output - dtype mismatch.
                    .replaces = GraphValueId{.index = 3},
            }},
            .op_params = ArgmaxParams{.axis = -1},
            .debug_name = "argmax_replacing_hidden",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsRedirectInducedDtypeMismatch) {
    // Phase B: ReplaceSubgraph must fail when a prior ReplaceValue redirects an
    // input to a value whose dtype is incompatible with the operator's slot
    // expectation. Here ReplaceValue(v2 -> v0) redirects the Float32 weight to
    // the Int32 tokens value; the Embedding weight slot then receives an Int32
    // spec, which InferOperator rejects.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    // Redirect v2 (Float32 weight) -> v0 (Int32 tokens_a).
    ASSERT_TRUE(session.ReplaceValue(GraphValueId{.index = 2},
                                     GraphValueId{.index = 0})
                        .ok());

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            // inputs[1] is v2, but GetResolvedValue(v2) == v0 (Int32).
            .inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 2}},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "redirected_hidden")},
            .op_params = EmbeddingParams{},
            .debug_name = "redirected_embed",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(GraphRewriteSession, ReplaceSubgraphRejectsUndefinedVirtualInputAtApplyTime) {
    // Phase B: ReplaceSubgraph must fail at apply time when an undefined
    // virtual value is fed to any input slot (here, the Embedding weight slot).
    // ResolveValueSpec returns NotFound for the unproduced virtual value.
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    const GraphValueId mid = session.AllocateVirtualValue();

    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            // mid is allocated but never produced by any replacement in this
            // group; the weight slot receives no inferred spec.
            .inputs = {GraphValueId{.index = 0}, mid},
            .outputs = {ReplacesHidden(GraphValueId{.index = 3}, "bad_hidden")},
            .op_params = EmbeddingParams{},
            .debug_name = "undefined_virtual_weight",
    };
    const std::array old_nodes{GraphNodeId{.index = 0}};
    const Status status = session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

}// namespace
}// namespace aethermind
