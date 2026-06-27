#include "aethermind/model/graph/graph_pass_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <string_view>
#include <vector>

namespace aethermind {
namespace {

TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

ModelGraph BuildGraph() {
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

class RedirectFirstNodeInputPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "RedirectFirstNodeInput";
    }

    Status Run(GraphRewriteSession& session) override {
        return session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1});
    }
};

class RemoveUnusedSecondNodePass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "RemoveUnusedSecondNode";
    }

    Status Run(GraphRewriteSession& session) override {
        return session.RemoveNode(GraphNodeId{.index = 1});
    }
};

class FailingPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "Failing";
    }

    Status Run(GraphRewriteSession&) override {
        return Status::InvalidArgument("intentional failure");
    }
};

TEST(GraphPassManager, EmptyPipelineReturnsValidGraph) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), graph.GetNodes().size());
    EXPECT_TRUE(result->Validate().ok());
}

TEST(GraphPassManager, RunsSinglePass) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<RedirectFirstNodeInputPass>());

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphNode& first_node = result->GetNode(GraphNodeId{.index = 0});
    EXPECT_EQ(first_node.inputs[0], result->GetInputs()[1].value);
}

TEST(GraphPassManager, StopsOnFirstError) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<FailingPass>())
            .Add(std::make_unique<RedirectFirstNodeInputPass>());

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(GraphPassManager, CheckpointCommitsIntermediateSnapshot) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(1)
            .Add(std::make_unique<RedirectFirstNodeInputPass>())
            .Add(std::make_unique<RemoveUnusedSecondNodePass>());

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

}// namespace
}// namespace aethermind
