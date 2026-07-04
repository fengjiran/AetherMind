#include "aethermind/model/graph/graph_pass_manager.h"

#include <gtest/gtest.h>
#include <memory>
#include <string_view>
#include <utility>
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

class RedirectFirstNodeInputPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "RedirectFirstNodeInput";
    }

    Status Run(GraphRewriteSession& session, const PassContext&) override {
        return session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1});
    }
};

class RemoveUnusedSecondNodePass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "RemoveUnusedSecondNode";
    }

    Status Run(GraphRewriteSession& session, const PassContext&) override {
        return session.RemoveNode(GraphNodeId{.index = 1});
    }
};

class FailingPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "Failing";
    }

    Status Run(GraphRewriteSession&, const PassContext&) override {
        return Status::InvalidArgument("intentional failure");
    }
};

class NoopPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "Noop";
    }

    Status Run(GraphRewriteSession&, const PassContext&) override {
        return Status::Ok();
    }
};

class ContextAwarePass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "ContextAware";
    }

    Status Run(GraphRewriteSession& session, const PassContext& ctx) override {
        if (!ctx.enable_qkv_fusion) {
            return Status::Ok();
        }
        return session.RedirectInput(GraphNodeId{.index = 0}, 0, GraphValueId{.index = 1});
    }
};

class ExpectCheckpointEveryPass final : public GraphPass {
public:
    explicit ExpectCheckpointEveryPass(uint32_t expected) noexcept
        : expected_(expected) {}

    AM_NODISCARD std::string_view Name() const noexcept override {
        return "ExpectCheckpointEvery";
    }

    Status Run(GraphRewriteSession&, const PassContext& ctx) override {
        if (ctx.checkpoint_every != expected_) {
            return Status::InvalidArgument("unexpected checkpoint_every");
        }
        return Status::Ok();
    }

private:
    uint32_t expected_ = 0;
};

template<typename... Passes>
std::vector<std::unique_ptr<GraphPass>> MakePasses(Passes&&... passes) {
    std::vector<std::unique_ptr<GraphPass>> result;
    result.reserve(sizeof...(passes));
    (result.push_back(std::forward<Passes>(passes)), ...);
    return result;
}

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
    pipeline.AddSequential(MakePasses(
            std::make_unique<FailingPass>(),
            std::make_unique<RedirectFirstNodeInputPass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(GraphPassManager, AddSequentialRunsPassesInOrder) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.AddSequential(MakePasses(
            std::make_unique<RedirectFirstNodeInputPass>(),
            std::make_unique<RemoveUnusedSecondNodePass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

TEST(GraphPassManager, AddSequentialSupportsChaining) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.AddSequential(MakePasses(std::make_unique<RedirectFirstNodeInputPass>()))
            .Add(std::make_unique<RemoveUnusedSecondNodePass>());

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

TEST(GraphPassManager, CheckpointCommitsIntermediateSnapshot) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(1)
            .AddSequential(MakePasses(
                    std::make_unique<RedirectFirstNodeInputPass>(),
                    std::make_unique<RemoveUnusedSecondNodePass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

// --- Issue P: SetCheckpointEvery(0) disables checkpointing ---

TEST(GraphPassManager, DisabledCheckpointStillProducesCorrectResult) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(0)// explicitly disable checkpointing
            .AddSequential(MakePasses(
                    std::make_unique<RedirectFirstNodeInputPass>(),
                    std::make_unique<RemoveUnusedSecondNodePass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // Both passes should still take effect via the final Commit()
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

// --- Issue I: checkpoint materializes on the last pass when it lands on
// a checkpoint boundary, instead of being skipped via a `!is_last` guard ---

TEST(GraphPassManager, CheckpointOnLastPassProducesCorrectResult) {
    // 3 passes with SetCheckpointEvery(3):
    //   pass 1 -> RedirectFirstNodeInput (mutates n0)
    //   pass 2 -> RemoveUnusedSecondNode (removes n1)
    //   pass 3 -> Noop (no mutation)
    // After pass 3, (i+1) % 3 == 0 lands on a checkpoint boundary AND pass 3
    // is the last pass. The pipeline must still materialize the snapshot
    // without relying on a redundant trailing Commit on the unchanged session.
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(3)
            .AddSequential(MakePasses(
                    std::make_unique<RedirectFirstNodeInputPass>(),
                    std::make_unique<RemoveUnusedSecondNodePass>(),
                    std::make_unique<NoopPass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

TEST(GraphPassManager, CheckpointBoundaryNotOnLastPassStillCommitsTrailing) {
    // 3 passes with SetCheckpointEvery(2):
    //   pass 1 -> RedirectFirstNodeInput
    //   pass 2 -> RemoveUnusedSecondNode (checkpoint fires here, i+1=2)
    //   pass 3 -> RedirectFirstNodeInput (trailing non-checkpoint pass)
    // Pass 3's mutation must be committed by the trailing Commit() call,
    // since (3 % 2) != 0 means pass 3 is not a checkpoint boundary.
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(2)
            .AddSequential(MakePasses(
                    std::make_unique<RedirectFirstNodeInputPass>(),
                    std::make_unique<RemoveUnusedSecondNodePass>(),
                    std::make_unique<RedirectFirstNodeInputPass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

TEST(GraphPassManager, ConstructorUsesCheckpointFromContext) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline(PassContext{.checkpoint_every = 1});
    pipeline.AddSequential(MakePasses(
            std::make_unique<RedirectFirstNodeInputPass>(),
            std::make_unique<RemoveUnusedSecondNodePass>()));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNode(GraphNodeId{.index = 0}).inputs[0], result->GetInputs()[1].value);
    EXPECT_TRUE(result->Validate().ok());
}

TEST(GraphPassManager, SetCheckpointEveryOverridesContext) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline(PassContext{.checkpoint_every = 3});
    pipeline.SetCheckpointEvery(1)
            .Add(std::make_unique<ExpectCheckpointEveryPass>(1));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
}

TEST(GraphPassManager, PassReceivesContext) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline(PassContext{.enable_qkv_fusion = false});
    pipeline.Add(std::make_unique<ContextAwarePass>());

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphNode& first_node = result->GetNode(GraphNodeId{.index = 0});
    EXPECT_EQ(first_node.inputs[0], result->GetInputs()[0].value);
}

TEST(GraphPassManager, RejectsNullPass) {
    const ModelGraph graph = BuildGraph();
    GraphPassManager pipeline;
    pipeline.AddSequential(MakePasses(
            std::make_unique<NoopPass>(),
            std::unique_ptr<GraphPass>{nullptr}));

    const StatusOr<ModelGraph> result = pipeline.Run(graph);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
