#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "test_optimization_helpers.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using namespace test_utils;

StatusOr<ModelGraph> RunDce(const ModelGraph& graph, PassContext ctx = {}) {
    GraphPassManager pipeline(ctx);
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    return pipeline.Run(graph);
}

class ReplaceAddWithSiluPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "ReplaceAddWithSiluPass";
    }

    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext&) const noexcept override {
        const std::vector<GraphNodeId> add_nodes = session.FindNodesByOpType(OpType::kAdd);
        AM_CHECK(add_nodes.size() == 1U, "Expected exactly one Add node");
        const GraphNodeId add_node = add_nodes[0];

        StatusOr<GraphNodeView> add_view = session.GetNodeView(add_node);
        AM_RETURN_IF_ERROR(add_view.status());
        auto output_desc_or = session.GetValueOutputMetadata(add_view->outputs[0]);
        AM_RETURN_IF_ERROR(output_desc_or.status());
        NodeOutputDesc output_desc{.payload = output_desc_or->payload,
                                   .quantization = output_desc_or->quantization,
                                   .name = output_desc_or->name};

        return session.ReplaceSubgraph(
                std::vector<GraphNodeId>{add_node},
                {{.op_type = OpType::kSilu,
                  .decoder_layer_index = add_view->decoder_layer_index,
                  .inputs = {add_view->inputs[0]},
                  .outputs = {RewriteOutputBinding{.desc = output_desc,
                                                   .replaces = add_view->outputs[0]}},
                  .op_params = SiluParams{},
                  .name = "replacement_silu"}});
    }
};

// Mimics a future value-replacement pass (CSE, algebraic simplification):
// replaces the directly-marked graph output (an Embedding output) with the
// output of an unrelated, consumer-less Silu node. At commit time the graph
// output resolves through the replacement chain to the Silu output.
class ReplaceGraphOutputWithSiluOutputPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "ReplaceGraphOutputWithSiluOutputPass";
    }

    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext&) const noexcept override {
        const std::vector<GraphNodeId> embedding_nodes = session.FindNodesByOpType(OpType::kEmbedding);
        const std::vector<GraphNodeId> silu_nodes = session.FindNodesByOpType(OpType::kSilu);
        if (silu_nodes.size() != 1U) {
            return Status::InvalidArgument("expected exactly one Silu node");
        }
        StatusOr<GraphNodeView> silu_view = session.GetNodeView(silu_nodes[0]);
        AM_RETURN_IF_ERROR(silu_view.status());

        bool found_marked_output = false;
        GraphValueId marked_output{};
        for (const GraphNodeId embedding_id: embedding_nodes) {
            StatusOr<GraphNodeView> embedding_view = session.GetNodeView(embedding_id);
            AM_RETURN_IF_ERROR(embedding_view.status());
            if (session.IsGraphOutput(embedding_view->outputs[0])) {
                marked_output = embedding_view->outputs[0];
                found_marked_output = true;
                break;
            }
        }
        if (!found_marked_output) {
            return Status::InvalidArgument("expected an Embedding node serving the graph output");
        }
        return session.ReplaceValue(marked_output, silu_view->outputs[0]);
    }
};

StatusOr<ModelGraph> RunReplaceOutputThenDce(const ModelGraph& graph) {
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<ReplaceGraphOutputWithSiluOutputPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    return pipeline.Run(graph);
}

StatusOr<ModelGraph> RunReplaceThenDce(const ModelGraph& graph) {
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<ReplaceAddWithSiluPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    return pipeline.Run(graph);
}

// Builds a two-hop replacement chain on the marked graph output:
// marked Embedding output -> Silu output -> Reorder output, so the terminal
// is the Reorder output and the Silu is an intermediate producer.
class ReplaceGraphOutputWithChainPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "ReplaceGraphOutputWithChainPass";
    }

    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext&) const noexcept override {
        const std::vector<GraphNodeId> embedding_nodes = session.FindNodesByOpType(OpType::kEmbedding);
        const std::vector<GraphNodeId> silu_nodes = session.FindNodesByOpType(OpType::kSilu);
        const std::vector<GraphNodeId> reorder_nodes = session.FindNodesByOpType(OpType::kReorder);
        if (silu_nodes.size() != 1U || reorder_nodes.size() != 1U) {
            return Status::InvalidArgument("expected exactly one Silu and one Reorder node");
        }
        StatusOr<GraphNodeView> silu_view = session.GetNodeView(silu_nodes[0]);
        AM_RETURN_IF_ERROR(silu_view.status());
        StatusOr<GraphNodeView> reorder_view = session.GetNodeView(reorder_nodes[0]);
        AM_RETURN_IF_ERROR(reorder_view.status());

        bool found_marked_output = false;
        GraphValueId marked_output{};
        for (const GraphNodeId embedding_id: embedding_nodes) {
            StatusOr<GraphNodeView> embedding_view = session.GetNodeView(embedding_id);
            AM_RETURN_IF_ERROR(embedding_view.status());
            if (session.IsGraphOutput(embedding_view->outputs[0])) {
                marked_output = embedding_view->outputs[0];
                found_marked_output = true;
                break;
            }
        }
        if (!found_marked_output) {
            return Status::InvalidArgument("expected an Embedding node serving the graph output");
        }
        AM_RETURN_IF_ERROR(session.ReplaceValue(marked_output, silu_view->outputs[0]));
        return session.ReplaceValue(silu_view->outputs[0], reorder_view->outputs[0]);
    }
};

StatusOr<ModelGraph> RunReplaceChainThenDce(const ModelGraph& graph) {
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<ReplaceGraphOutputWithChainPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    return pipeline.Run(graph);
}

TEST(DeadCodeEliminationPass, RemovesUnusedNode) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    AddActivation(graph, "dead");
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNodes()[0].name, "live");
}

TEST(DeadCodeEliminationPass, SkipsWhenDisabled) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    AddActivation(graph, "dead");
    graph.MarkOutput(live);
    PassContext ctx;
    ctx.enable_dce = false;

    const StatusOr<ModelGraph> result = RunDce(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 2U);
}

TEST(DeadCodeEliminationPass, RemovesDeadChain) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId dead_input = AddActivation(graph, "dead_input");
    auto dead_silu_or = AddSilu(graph, 0U, dead_input, "dead_silu");
    ASSERT_TRUE(dead_silu_or.ok()) << dead_silu_or.status().ToString();
    const GraphValueId dead_silu = *dead_silu_or;
    auto dead_mul_or = AddElementwiseMul(graph, 0U,
                                         dead_silu, dead_input, "dead_mul");
    ASSERT_TRUE(dead_mul_or.ok()) << dead_mul_or.status().ToString();
    const GraphValueId dead_mul = *dead_mul_or;
    UNUSED(dead_mul);
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
}

TEST(DeadCodeEliminationPass, KeepsGraphOutputProducer) {
    ModelGraph graph;
    const GraphValueId hidden = AddActivation(graph, "hidden");
    graph.MarkOutput(hidden);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsProducerWithLiveConsumer) {
    ModelGraph graph;
    const GraphValueId lhs = AddActivation(graph, "lhs");
    const GraphValueId rhs = AddActivation(graph, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 3U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsMultiOutputNodeWhenAnyOutputIsGraphOutput) {
    ModelGraph graph;
    const GraphValueId q = AddActivation(graph, "q");
    const GraphValueId k = AddActivation(graph, "k");
    const GraphValueId position_ids = graph.AddInput(
            Spec(DataType::Int(64), {2}), "position_ids");
    auto rope_or = AddRoPE(graph,
                           0U,
                           q,
                           k,
                           position_ids,
                           RoPEParams{.head_dim = 4,
                                      .num_attention_heads = 1,
                                      .num_key_value_heads = 1,
                                      .max_position_embeddings = 128,
                                      .theta = 10000.0,
                                      .scaling_type = RoPEScalingType::kNone},
                           "rope");
    ASSERT_TRUE(rope_or.ok()) << rope_or.status().ToString();
    const RoPEOutputs rope = *rope_or;
    graph.MarkOutput(rope.q);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kRoPE).size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsStateOutputNodeWithoutConsumers) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId k_new = AddActivation(graph, "k_new");
    const GraphValueId v_new = AddActivation(graph, "v_new");
    const GraphValueId k_cache = AddState(graph,
                                          Spec(DataType::Float32(), {1, 2, 4}),
                                          KVCacheStateBinding{
                                                  .decoder_layer_index = 0,
                                                  .slot = KVCacheSlot::kKey},
                                          "k_cache");
    const GraphValueId v_cache = AddState(graph,
                                          Spec(DataType::Float32(), {1, 2, 4}),
                                          KVCacheStateBinding{
                                                  .decoder_layer_index = 0,
                                                  .slot = KVCacheSlot::kValue},
                                          "v_cache");
    auto updated_cache_or = AddKVCacheUpdate(
            graph, 0U, k_new, v_new, k_cache, v_cache, "kv_update");
    ASSERT_TRUE(updated_cache_or.ok()) << updated_cache_or.status().ToString();
    const KVCachePair updated_cache = *updated_cache_or;
    UNUSED(updated_cache);
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kKVCacheUpdate).size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsProducerConsumedOnlyByActiveReplacement) {
    ModelGraph graph;
    const GraphValueId lhs = AddActivation(graph, "lhs");
    const GraphValueId rhs = AddActivation(graph, "rhs");
    auto sum_or = AddElementwiseAdd(
            graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);

    const StatusOr<ModelGraph> result = RunReplaceThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kEmbedding).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsTerminalProducerOfReplacedGraphOutput) {
    // Regression test: a pass replaces the directly-marked graph output (a)
    // with the output of a consumer-less, unmarked node (b_silu). Commit
    // resolves the graph output to b's output terminal (MarkCommittedOutputs),
    // so the b producer must survive DCE. Removing it previously failed the
    // whole pipeline with an unmappable output ("producer removed or not yet
    // emitted").
    ModelGraph graph;
    const GraphValueId a_out = AddActivation(graph, "a");
    const GraphValueId b_input = AddActivation(graph, "b_input");
    auto b_silu_or = AddSilu(graph, 0U, b_input, "b_silu");
    ASSERT_TRUE(b_silu_or.ok()) << b_silu_or.status().ToString();
    graph.MarkOutput(a_out);

    const StatusOr<ModelGraph> result = RunReplaceOutputThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    const std::vector<GraphNodeId> silu_nodes = result->FindNodesByOpType(OpType::kSilu);
    ASSERT_EQ(silu_nodes.size(), 1U);
    const GraphNode& silu_node = result->GetNode(silu_nodes[0]);
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    EXPECT_EQ(result->GetOutputs()[0].value, silu_node.outputs[0]);
}

TEST(DeadCodeEliminationPass, KeepsTerminalProducerOfReplacedGraphOutputChain) {
    // Chain variant: marked graph output resolves through two replacement
    // hops (a -> silu -> reorder). The intermediate Silu producer and the
    // original Embedding producer must be removed, while the terminal
    // Reorder producer must survive and carry the committed graph output.
    ModelGraph graph;
    const GraphValueId a_out = AddActivation(graph, "a");
    const GraphValueId b_input = AddActivation(graph, "b_input");
    auto b_silu_or = AddSilu(graph, 0U, b_input, "b_silu");
    ASSERT_TRUE(b_silu_or.ok()) << b_silu_or.status().ToString();
    const GraphValueId c_input = AddActivation(graph, "c_input");
    auto c_reorder_or = AddReorder(graph, std::nullopt, c_input, "c_reorder");
    ASSERT_TRUE(c_reorder_or.ok()) << c_reorder_or.status().ToString();
    graph.MarkOutput(a_out);

    const StatusOr<ModelGraph> result = RunReplaceChainThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    const std::vector<GraphNodeId> reorder_nodes = result->FindNodesByOpType(OpType::kReorder);
    ASSERT_EQ(reorder_nodes.size(), 1U);
    const GraphNode& reorder_node = result->GetNode(reorder_nodes[0]);
    ASSERT_EQ(result->GetOutputs().size(), 1U);
    EXPECT_EQ(result->GetOutputs()[0].value, reorder_node.outputs[0]);
}

TEST(DeadCodeEliminationPass, IsIdempotent) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    AddActivation(graph, "dead");
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> first = RunDce(graph);
    ASSERT_TRUE(first.ok()) << first.status().ToString();
    const StatusOr<ModelGraph> second = RunDce(*first);

    ASSERT_TRUE(second.ok()) << second.status().ToString();
    ASSERT_TRUE(second->Validate().ok());
    EXPECT_EQ(second->GetNodes().size(), first->GetNodes().size());
    EXPECT_EQ(second->GetValues().size(), first->GetValues().size());
}

// ── Rank-zero DCE: dead rank-0 arithmetic nodes must be removed ──

namespace {

GraphValueId AddRankZeroConstantFloat(ModelGraph& graph, float value, const std::string& name) {
    return graph.AddConstant(
            Spec(DataType::Float32(), {}),
            ConstantBinding{.inline_data = InlineFloats({value}), .name = name},
            name);
}

} // namespace

TEST(DeadCodeEliminationPass, RemovesDeadRankZeroAdd) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId lhs = AddRankZeroConstantFloat(graph, 1.0F, "lhs");
    const GraphValueId rhs = AddRankZeroConstantFloat(graph, 2.0F, "rhs");
    auto dead_sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "dead_sum");
    ASSERT_TRUE(dead_sum_or.ok()) << dead_sum_or.status().ToString();
    const GraphValueId dead_sum = *dead_sum_or;
    UNUSED(dead_sum);
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
}

TEST(DeadCodeEliminationPass, RemovesDeadRankZeroSilu) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId input = AddRankZeroConstantFloat(graph, 3.0F, "input");
    auto dead_act_or = AddSilu(graph, 0U, input, "dead_silu");
    ASSERT_TRUE(dead_act_or.ok()) << dead_act_or.status().ToString();
    const GraphValueId dead_act = *dead_act_or;
    UNUSED(dead_act);
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
}

TEST(DeadCodeEliminationPass, RemovesDeadRankZeroSiluMul) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId gate = AddRankZeroConstantFloat(graph, 1.0F, "gate");
    const GraphValueId up = AddRankZeroConstantFloat(graph, 2.0F, "up");
    auto dead_act_or = AddSiluMul(graph, 0U, gate, up, "dead_silu_mul");
    ASSERT_TRUE(dead_act_or.ok()) << dead_act_or.status().ToString();
    const GraphValueId dead_act = *dead_act_or;
    UNUSED(dead_act);
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

TEST(DeadCodeEliminationPass, LiveReorderSurvivesDCE) {
    // A Reorder whose output is a graph output must survive DCE.
    ModelGraph graph;
    const GraphValueId input = AddActivation(graph, "input");
    auto reorder_or = AddReorder(graph, std::nullopt, input, "reorder");
    ASSERT_TRUE(reorder_or.ok()) << reorder_or.status().ToString();
    graph.MarkOutput(*reorder_or);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kReorder).size(), 1U);
}

TEST(DeadCodeEliminationPass, DeadReorderIsRemoved) {
    // A Reorder whose output is unused must be removed by DCE.
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId input = AddActivation(graph, "input");
    auto reorder_or = AddReorder(graph, std::nullopt, input, "dead_reorder");
    ASSERT_TRUE(reorder_or.ok()) << reorder_or.status().ToString();
    const GraphValueId dead = *reorder_or;
    UNUSED(dead);
    graph.MarkOutput(live);

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kReorder).size(), 0U);
}

} // namespace
