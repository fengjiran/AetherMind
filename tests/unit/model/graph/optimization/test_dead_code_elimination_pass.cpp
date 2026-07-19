#include "../test_graph_helpers.h"
#include "aethermind/model/graph/graph_op_builder.h"
#include "aethermind/model/graph/optimization/dead_code_elimination_pass.h"

#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace aethermind {
namespace {

GraphValueId AddActivation(ModelGraph& graph, const char* debug_name) {
    const GraphValueId tokens = graph.AddInput(
            Spec(DataType::Int(32), {2}), std::string(debug_name) + ".tokens");
    return AddEmbedding(graph,
                        tokens,
                        16,
                        4,
                        DataType::Float32(),
                        WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                      .semantic_role = TransformerWeightRole::kTokenEmbedding},
                        debug_name);
}

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

    AM_NODISCARD Status Run(GraphRewriteSession& session, const PassContext&) override {
        const std::vector<GraphNodeId> add_nodes = session.FindNodesByOpType(OpType::kAdd);
        AM_CHECK(add_nodes.size() == 1U, "Expected exactly one Add node");
        const GraphNodeId add_node = add_nodes[0];

        StatusOr<GraphNodeView> add_view = session.GetNodeView(add_node);
        AM_RETURN_IF_ERROR(add_view.status());
        auto output_desc_or = session.GetValueOutputDesc(add_view->outputs[0]);
        AM_RETURN_IF_ERROR(output_desc_or.status());
        NodeOutputDesc output_desc{.payload = output_desc_or->payload,
                                   .quantization = output_desc_or->quantization,
                                   .debug_name = output_desc_or->debug_name};

        return session.ReplaceSubgraph(
                std::vector<GraphNodeId>{add_node},
                {{.op_type = OpType::kSilu,
                  .decoder_layer_index = add_view->decoder_layer_index,
                  .inputs = {add_view->inputs[0]},
                  .outputs = {RewriteOutputBinding{.desc = output_desc,
                                                   .replaces = add_view->outputs[0]}},
                  .op_params = SiluParams{},
                  .debug_name = "replacement_silu"}});
    }
};

StatusOr<ModelGraph> RunReplaceThenDce(const ModelGraph& graph) {
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<ReplaceAddWithSiluPass>());
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    return pipeline.Run(graph);
}

TEST(DeadCodeEliminationPass, RemovesUnusedNode) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    AddActivation(graph, "dead");
    graph.MarkOutput(live, "output");

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 1U);
    EXPECT_EQ(result->GetNodes()[0].debug_name, "live");
}

TEST(DeadCodeEliminationPass, SkipsWhenDisabled) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    AddActivation(graph, "dead");
    graph.MarkOutput(live, "output");
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
    const GraphValueId dead_silu = AddSilu(graph, 0U, dead_input, "dead_silu");
    const GraphValueId dead_mul = AddElementwiseMul(graph, 0U,
                                                    dead_silu, dead_input, "dead_mul");
    UNUSED(dead_mul);
    graph.MarkOutput(live, "output");

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
    graph.MarkOutput(hidden, "output");

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->GetNodes().size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsProducerWithLiveConsumer) {
    ModelGraph graph;
    const GraphValueId lhs = AddActivation(graph, "lhs");
    const GraphValueId rhs = AddActivation(graph, "rhs");
    const GraphValueId sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

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
    const RoPEOutputs rope = AddRoPE(graph,
                                     0U,
                                     q,
                                     k,
                                     position_ids,
                                     RoPEParams{.head_dim = 4,
                                                .num_attention_heads = 1,
                                                .num_key_value_heads = 1,
                                                .max_position_embeddings = 128,
                                                .theta = 10000.0,
                                                .scaling_type = HfRopeScalingType::kNone},
                                     "rope");
    graph.MarkOutput(rope.q, "q_rope");

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
                                          Spec(DataType::Float32(), {2, 4}),
                                          KVCacheStateBinding{
                                                  .decoder_layer_index = 0,
                                                  .slot = KVCacheSlot::kKey},
                                          "k_cache");
    const GraphValueId v_cache = AddState(graph,
                                          Spec(DataType::Float32(), {2, 4}),
                                          KVCacheStateBinding{
                                                  .decoder_layer_index = 0,
                                                  .slot = KVCacheSlot::kValue},
                                          "v_cache");
    const KVCachePair updated_cache = AddKVCacheUpdate(
            graph, 0U, k_new, v_new, k_cache, v_cache, "kv_update");
    UNUSED(updated_cache);
    graph.MarkOutput(live, "output");

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kKVCacheUpdate).size(), 1U);
}

TEST(DeadCodeEliminationPass, KeepsProducerConsumedOnlyByActiveReplacement) {
    ModelGraph graph;
    const GraphValueId lhs = AddActivation(graph, "lhs");
    const GraphValueId rhs = AddActivation(graph, "rhs");
    const GraphValueId sum = AddElementwiseAdd(
            graph, 0U, lhs, rhs, "sum");
    graph.MarkOutput(sum, "output");

    const StatusOr<ModelGraph> result = RunReplaceThenDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kEmbedding).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
}

TEST(DeadCodeEliminationPass, IsIdempotent) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    AddActivation(graph, "dead");
    graph.MarkOutput(live, "output");

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

std::shared_ptr<const std::vector<std::byte>> MakeBytes(std::vector<float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_shared<const std::vector<std::byte>>(std::move(bytes));
}

GraphValueId AddRankZeroConstantFloat(ModelGraph& graph, float value, const std::string& name) {
    return graph.AddConstant(
            Spec(DataType::Float32(), {}),
            ConstantBinding{.inline_data = MakeBytes({value}), .name = name},
            name);
}

}// namespace

TEST(DeadCodeEliminationPass, RemovesDeadRankZeroAdd) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId lhs = AddRankZeroConstantFloat(graph, 1.0F, "lhs");
    const GraphValueId rhs = AddRankZeroConstantFloat(graph, 2.0F, "rhs");
    const GraphValueId dead_sum = AddElementwiseAdd(graph, 0U, lhs, rhs, "dead_sum");
    UNUSED(dead_sum);
    graph.MarkOutput(live, "output");

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kAdd).size(), 0U);
}

TEST(DeadCodeEliminationPass, RemovesDeadRankZeroSilu) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId input = AddRankZeroConstantFloat(graph, 3.0F, "input");
    const GraphValueId dead_act = AddSilu(graph, 0U, input, "dead_silu");
    UNUSED(dead_act);
    graph.MarkOutput(live, "output");

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
    const GraphValueId dead_act = AddSiluMul(graph, 0U, gate, up, "dead_silu_mul");
    UNUSED(dead_act);
    graph.MarkOutput(live, "output");

    const StatusOr<ModelGraph> result = RunDce(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

}// namespace
}// namespace aethermind
