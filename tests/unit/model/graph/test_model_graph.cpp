#include "aethermind/model/graph/model_graph.h"

#include <gtest/gtest.h>

#include <span>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

TensorSpec Spec(DataType dtype, const std::vector<int64_t>& dims) {
    std::vector<ShapeSymbol> symbols;
    symbols.reserve(dims.size());
    for (const int64_t dim: dims) {
        symbols.push_back(ShapeSymbol::CreateFromValue(dim));
    }
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(std::move(symbols))};
}

TensorSpec ActivationSpec() {
    return Spec(DataType::Float32(), {1, 8});
}

TensorSpec TokenSpec() {
    return Spec(DataType::Int(64), {1});
}

TensorSpec WeightSpec() {
    return Spec(DataType::Float32(), {8});
}

ModelGraph BuildValidEmbeddingGraph() {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), ModelWeightBinding{.role = ModelWeightRole::kTokenEmbedding});
    const ModelGraph::AddedNode embedding = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            EmbeddingParams{});
    graph.MarkOutput(embedding.outputs[0], "hidden");
    return graph;
}

TEST(ModelGraph, PublicApiCreatesInputsWeightsNodesAndOutputs) {
    ModelGraph graph;

    const GraphValueId input = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), ModelWeightBinding{.role = ModelWeightRole::kInputNorm});
    const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}, .debug_name = "normed"}},
            RmsNormParams{},
            {},
            "rms_norm_0");
    graph.MarkOutput(node.outputs[0], "normed_output");

    ASSERT_EQ(graph.GetInputs().size(), 1U);
    EXPECT_EQ(graph.GetInputs()[0].value, input);
    EXPECT_EQ(graph.GetInputs()[0].name, "token_ids");
    ASSERT_EQ(graph.GetOutputs().size(), 1U);
    EXPECT_EQ(graph.GetOutputs()[0].value, node.outputs[0]);
    EXPECT_EQ(graph.GetOutputs()[0].name, "normed_output");
    ASSERT_EQ(graph.GetNodes().size(), 1U);
    EXPECT_EQ(graph.GetNode(node.node).op_type, OpType::kRmsNorm);
    EXPECT_EQ(graph.GetNode(node.node).debug_name, "rms_norm_0");
    ASSERT_EQ(graph.GetValues().size(), 3U);
    EXPECT_TRUE(std::holds_alternative<ModelInputValue>(graph.GetValue(input).payload));
    EXPECT_TRUE(std::holds_alternative<WeightValue>(graph.GetValue(weight).payload));
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(graph.GetValue(node.outputs[0]).payload));
    ASSERT_TRUE(graph.GetValue(node.outputs[0]).producer.has_value());
    EXPECT_EQ(*graph.GetValue(node.outputs[0]).producer, node.node);
}

TEST(ModelGraph, ValidateAcceptsValidGraph) {
    const ModelGraph graph = BuildValidEmbeddingGraph();

    EXPECT_TRUE(graph.Validate().ok());
}

TEST(ModelGraph, OpParamsRoundTripThroughAddNode) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), ModelWeightBinding{.role = ModelWeightRole::kInputNorm});

    const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RmsNormParams{.eps = 2.5e-3F});

    const auto* params = std::get_if<RmsNormParams>(&graph.GetNode(node.node).op_params);
    ASSERT_NE(params, nullptr);
    EXPECT_FLOAT_EQ(params->eps, 2.5e-3F);
}

TEST(ModelGraph, ValidateRejectsMissingOpParams) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), ModelWeightBinding{.role = ModelWeightRole::kTokenEmbedding});
    [[maybe_unused]] const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsWrongOpParamsForOpType) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), ModelWeightBinding{.role = ModelWeightRole::kTokenEmbedding});
    [[maybe_unused]] const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RmsNormParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidRmsNormEps) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), ModelWeightBinding{.role = ModelWeightRole::kInputNorm});
    [[maybe_unused]] const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RmsNormParams{.eps = 0.0F});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidRoPEParams) {
    ModelGraph graph;
    const GraphValueId q = graph.AddInput(ActivationSpec(), "q");
    const GraphValueId k = graph.AddInput(ActivationSpec(), "k");
    [[maybe_unused]] const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kRoPE,
            0U,
            {q, k},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}},
             ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RoPEParams{.head_dim = 0,
                       .num_attention_heads = 4,
                       .num_key_value_heads = 2,
                       .max_position_embeddings = 128});

    const Status status = graph.Validate();
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsMonostatePayload) {
    ModelGraph graph({}, {}, {GraphValue{}});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsActivationWithoutProducer) {
    ModelGraph graph({}, {}, {GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec()}});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsExternalValueWithProducer) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec(), .producer = GraphNodeId{0}},
    };
    ModelGraph graph({}, {}, std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsSchemaPortKindMismatch) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec()},
            GraphValue{.payload = ActivationValue{}, .spec = WeightSpec(), .producer = GraphNodeId{0}},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kEmbedding, .inputs = {GraphValueId{0}, GraphValueId{1}}, .outputs = {GraphValueId{2}}},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsSchemaArityMismatch) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec()},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kEmbedding, .inputs = {GraphValueId{0}}, .outputs = {GraphValueId{1}}},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsOutputOnExternalValue) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), ModelWeightBinding{.role = ModelWeightRole::kInputNorm});
    const ModelGraph::AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}});
    graph.MarkOutput(input, "bad_output");

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, BuildConsumerIndexMapsConsumers) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const ModelGraph::AddedNode softmax = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}});
    const ModelGraph::AddedNode argmax = graph.AddNode(
            OpType::kArgmax,
            std::nullopt,
            {softmax.outputs[0]},
            {ModelGraph::NodeOutputDecl{.spec = TokenSpec(), .payload = ActivationValue{}}});
    const ModelGraph::AddedNode another = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {softmax.outputs[0]},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}});

    const StatusOr<std::vector<std::vector<GraphNodeId>>> index = BuildConsumerIndex(graph);

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    ASSERT_EQ(index->size(), graph.GetValues().size());
    const std::span<const GraphNodeId> input_consumers = GetConsumers(*index, input);
    ASSERT_EQ(input_consumers.size(), 1U);
    EXPECT_EQ(input_consumers[0], softmax.node);
    const std::span<const GraphNodeId> softmax_consumers = GetConsumers(*index, softmax.outputs[0]);
    ASSERT_EQ(softmax_consumers.size(), 2U);
    EXPECT_EQ(softmax_consumers[0], argmax.node);
    EXPECT_EQ(softmax_consumers[1], another.node);
    EXPECT_TRUE(GetConsumers(*index, argmax.outputs[0]).empty());
}

TEST(ModelGraph, GetConsumersReturnsEmptyForInvalidValueId) {
    const std::vector<std::vector<GraphNodeId>> index(4);
    EXPECT_TRUE(GetConsumers(index, GraphValueId{10}).empty());
}

TEST(ModelGraph, TopologicalOrderReturnsStableProducerBeforeConsumerOrder) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const ModelGraph::AddedNode softmax = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {ModelGraph::NodeOutputDecl{.spec = ActivationSpec(), .payload = ActivationValue{}}});
    const ModelGraph::AddedNode argmax = graph.AddNode(
            OpType::kArgmax,
            std::nullopt,
            {softmax.outputs[0]},
            {ModelGraph::NodeOutputDecl{.spec = TokenSpec(), .payload = ActivationValue{}}});

    const StatusOr<std::vector<GraphNodeId>> order = graph.TopologicalOrder();

    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), 2U);
    EXPECT_EQ((*order)[0], softmax.node);
    EXPECT_EQ((*order)[1], argmax.node);
}

TEST(ModelGraph, TopologicalOrderRejectsActivationCycle) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{1}},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kSoftmax, .inputs = {GraphValueId{1}}, .outputs = {GraphValueId{0}}},
            GraphNode{.op_type = OpType::kSoftmax, .inputs = {GraphValueId{0}}, .outputs = {GraphValueId{1}}},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const StatusOr<std::vector<GraphNodeId>> order = graph.TopologicalOrder();

    ASSERT_FALSE(order.ok());
    EXPECT_EQ(order.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
