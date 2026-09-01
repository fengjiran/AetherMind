#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/graph/optimization/graph_pass_manager.h"
#include "aethermind/graph/optimization/graph_rewrite.h"
#include "test_optimization_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::test_utils;

NodeOutputDesc ActivationDesc(std::string name) {
    return {.payload = ActivationValue{}, .name = std::move(name)};
}

RewriteOutputBinding Replaces(GraphValueId value, std::string name) {
    return {.desc = ActivationDesc(std::move(name)), .replaces = value};
}

GraphNodeId ProducerOf(const ModelGraph& graph, GraphValueId value) {
    const std::optional<GraphNodeId> producer = graph.GetValue(value).producer;
    AM_CHECK(producer.has_value(), "expected value producer");
    return *producer;
}

ReplacementNode SiluReplacement(GraphValueId input,
                                std::optional<GraphValueId> replaces,
                                std::string name) {
    return {.op_type = OpType::kSilu,
            .inputs = {input},
            .outputs = {{.desc = ActivationDesc(name + ".output"),
                         .replaces = replaces}},
            .op_params = SiluParams{},
            .name = std::move(name)};
}

CommitOptions ForcePruning() {
    return {.force_prune_unreachable = true};
}

bool HasNodeNamed(const ModelGraph& graph, const std::string& name) {
    return std::any_of(graph.GetNodes().begin(),
                       graph.GetNodes().end(),
                       [&](const GraphNode& node) {
                           return node.name == name;
                       });
}

bool HasValueNamed(const ModelGraph& graph, const std::string& name) {
    return std::any_of(graph.GetValues().begin(),
                       graph.GetValues().end(),
                       [&](const GraphValue& value) {
                           return value.name == name;
                       });
}

void ExpectSameGraphStructure(const ModelGraph& lhs, const ModelGraph& rhs) {
    ASSERT_EQ(lhs.GetNodes().size(), rhs.GetNodes().size());
    ASSERT_EQ(lhs.GetValues().size(), rhs.GetValues().size());
    ASSERT_EQ(lhs.GetOutputs().size(), rhs.GetOutputs().size());
    for (std::size_t i = 0; i < lhs.GetNodes().size(); ++i) {
        EXPECT_EQ(lhs.GetNodes()[i].op_type, rhs.GetNodes()[i].op_type);
        EXPECT_EQ(lhs.GetNodes()[i].name, rhs.GetNodes()[i].name);
        EXPECT_EQ(lhs.GetNodes()[i].inputs, rhs.GetNodes()[i].inputs);
        EXPECT_EQ(lhs.GetNodes()[i].outputs, rhs.GetNodes()[i].outputs);
    }
    for (std::size_t i = 0; i < lhs.GetOutputs().size(); ++i) {
        EXPECT_EQ(lhs.GetOutputs()[i].value, rhs.GetOutputs()[i].value);
    }
}

class NoopPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "CommitPruningNoopPass";
    }

    Status Run(GraphRewriteSession&, const PassContext&) const noexcept override {
        return Status::Ok();
    }
};

class InstallOrphanReplacementPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "InstallOrphanReplacementPass";
    }

    Status Run(GraphRewriteSession& session, const PassContext&) const noexcept override {
        const GraphNodeId dead_node{.index = 1};
        AM_ASSIGN_OR_RETURN(const GraphNodeView view, session.GetNodeView(dead_node));
        AM_ASSIGN_OR_RETURN(const GraphValueDesc output,
                            session.GetValueOutputMetadata(view.outputs[0]));
        ReplacementNode replacement{
                .op_type = OpType::kEmbedding,
                .inputs = view.inputs,
                .outputs = {{.desc = {.payload = output.payload,
                                      .quantization = output.quantization,
                                      .name = output.name},
                             .replaces = view.outputs[0]}},
                .op_params = EmbeddingParams{},
                .name = "orphan_replacement"};
        const std::array old_nodes{dead_node};
        return session.ReplaceSubgraph(old_nodes, {std::move(replacement)});
    }
};

class ExpectOrphanReplacementPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "ExpectOrphanReplacementPass";
    }

    Status Run(GraphRewriteSession& session, const PassContext&) const noexcept override {
        for (const GraphNodeId node_id: session.FindNodesByOpType(OpType::kEmbedding)) {
            AM_ASSIGN_OR_RETURN(const GraphNodeView view, session.GetNodeView(node_id));
            if (view.name == "orphan_replacement") {
                return Status::Ok();
            }
        }
        return Status::InvalidArgument("orphan replacement was pruned before DCE ran");
    }
};

ModelGraph BuildConstantRewriteGraph() {
    ModelGraph graph;
    const GraphValueId live_input = AddActivation(graph, "live_input");
    const GraphValueId dead_input = AddActivation(graph, "dead_input");
    auto live_or = AddSilu(graph, 0U, live_input, "live_target");
    auto dead_or = AddSilu(graph, 0U, dead_input, "dead_target");
    AM_CHECK(live_or.ok(), "live target AddSilu failed");
    AM_CHECK(dead_or.ok(), "dead target AddSilu failed");
    graph.MarkOutput(*live_or);
    return graph;
}

class InstallConstantRewritesPass final : public GraphPass {
public:
    AM_NODISCARD std::string_view Name() const noexcept override {
        return "InstallConstantRewritesPass";
    }

    Status Run(GraphRewriteSession& session, const PassContext&) const noexcept override {
        for (const GraphNodeId node_id: session.FindNodesByOpType(OpType::kSilu)) {
            AM_ASSIGN_OR_RETURN(const GraphNodeView view, session.GetNodeView(node_id));
            const bool retained = session.IsGraphOutput(view.outputs[0]);
            const std::string prefix = retained ? "retained" : "pruned";
            AM_ASSIGN_OR_RETURN(const GraphValueDesc output,
                                session.GetValueOutputMetadata(view.outputs[0]));
            const GraphValueId constant = session.AddSessionConstant(
                    output.spec,
                    ConstantBinding{.inline_data = InlineFloats(std::vector<float>(8, 1.0F)),
                                    .name = prefix + ".constant"},
                    {},
                    prefix + ".constant");
            ReplacementNode replacement =
                    SiluReplacement(constant, view.outputs[0], prefix + "_replacement");
            const std::array old_nodes{node_id};
            AM_RETURN_IF_ERROR(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}));
        }
        return Status::Ok();
    }
};

TEST(CommitPruning, PrunesOrphanReplacement) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    ReplacementNode replacement{
            .op_type = OpType::kEmbedding,
            .inputs = {GraphValueId{.index = 1}, GraphValueId{.index = 2}},
            .outputs = {Replaces(GraphValueId{.index = 4}, "orphan.output")},
            .op_params = EmbeddingParams{},
            .name = "orphan_replacement"};
    const std::array old_nodes{GraphNodeId{.index = 1}};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_FALSE(HasNodeNamed(*committed, "orphan_replacement"));
}

TEST(CommitPruning, PrunedReplacementDoesNotKeepSourceProducerAlive) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId produced = AddActivation(graph, "source_producer");
    auto target_or = AddSilu(graph, 0U, produced, "dead_target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    graph.MarkOutput(live);
    GraphRewriteSession session(graph);
    const GraphNodeId target = ProducerOf(graph, *target_or);
    const std::array old_nodes{target};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(produced, *target_or, "orphan_replacement")})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_FALSE(HasNodeNamed(*committed, "source_producer"));
    EXPECT_FALSE(HasNodeNamed(*committed, "orphan_replacement"));
}

TEST(CommitPruning, PrunesUpstreamOfReplacedSourceProducer) {
    ModelGraph graph;
    const GraphValueId upstream = AddActivation(graph, "upstream");
    auto replaced_or = AddSilu(graph, 0U, upstream, "replaced_source");
    ASSERT_TRUE(replaced_or.ok()) << replaced_or.status().ToString();
    const GraphValueId replacement_input = AddActivation(graph, "replacement_input");
    graph.MarkOutput(*replaced_or);
    GraphRewriteSession session(graph);
    const std::array old_nodes{ProducerOf(graph, *replaced_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(replacement_input,
                                                *replaced_or,
                                                "retained_replacement")})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_FALSE(HasNodeNamed(*committed, "upstream"));
    EXPECT_FALSE(HasNodeNamed(*committed, "replaced_source"));
    EXPECT_TRUE(HasNodeNamed(*committed, "replacement_input"));
    EXPECT_TRUE(HasNodeNamed(*committed, "retained_replacement"));
}

TEST(CommitPruning, KeepsReplacementConsumedByLiveNode) {
    ModelGraph graph;
    const GraphValueId input = AddActivation(graph, "input");
    auto target_or = AddSilu(graph, 0U, input, "target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    auto consumer_or = AddSilu(graph, 0U, *target_or, "live_consumer");
    ASSERT_TRUE(consumer_or.ok()) << consumer_or.status().ToString();
    graph.MarkOutput(*consumer_or);
    GraphRewriteSession session(graph);
    const std::array old_nodes{ProducerOf(graph, *target_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(input, *target_or, "retained_replacement")})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 3U);
    EXPECT_TRUE(HasNodeNamed(*committed, "retained_replacement"));
    EXPECT_TRUE(HasNodeNamed(*committed, "live_consumer"));
}

TEST(CommitPruning, KeepsReplacementTerminalForGraphOutput) {
    ModelGraph graph;
    const GraphValueId input = AddActivation(graph, "input");
    auto target_or = AddSilu(graph, 0U, input, "target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    graph.MarkOutput(*target_or);
    GraphRewriteSession session(graph);
    const std::array old_nodes{ProducerOf(graph, *target_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(input, *target_or, "terminal_replacement")})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_TRUE(HasNodeNamed(*committed, "terminal_replacement"));
}

TEST(CommitPruning, PrunesEntireUnreachableChain) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId chain_input = AddActivation(graph, "chain_input");
    auto target_or = AddSilu(graph, 0U, chain_input, "dead_target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    graph.MarkOutput(live);
    GraphRewriteSession session(graph);
    const GraphValueId middle = session.AllocateVirtualValue();
    const std::array old_nodes{ProducerOf(graph, *target_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(chain_input, middle, "chain_head"),
                                SiluReplacement(middle, *target_or, "chain_tail")})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_FALSE(HasNodeNamed(*committed, "chain_input"));
    EXPECT_FALSE(HasNodeNamed(*committed, "chain_head"));
    EXPECT_FALSE(HasNodeNamed(*committed, "chain_tail"));
}

TEST(CommitPruning, KeepsDependencyChainForLiveTail) {
    ModelGraph graph;
    const GraphValueId chain_input = AddActivation(graph, "chain_input");
    auto target_or = AddSilu(graph, 0U, chain_input, "target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    graph.MarkOutput(*target_or);
    GraphRewriteSession session(graph);
    const GraphValueId middle = session.AllocateVirtualValue();
    const std::array old_nodes{ProducerOf(graph, *target_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(chain_input, middle, "chain_head"),
                                SiluReplacement(middle, *target_or, "chain_tail")})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 3U);
    EXPECT_TRUE(HasNodeNamed(*committed, "chain_head"));
    EXPECT_TRUE(HasNodeNamed(*committed, "chain_tail"));
}

TEST(CommitPruning, KeepsLivePrefixAndPrunesDeadSuffix) {
    ModelGraph graph;
    const GraphValueId q = AddActivation(graph, "q");
    const GraphValueId k = AddActivation(graph, "k");
    const GraphValueId position = graph.AddInput(Spec(DataType::Int(64), {2}), "position");
    const RoPEParams params{.head_dim = 4,
                            .num_attention_heads = 1,
                            .num_key_value_heads = 1,
                            .max_position_embeddings = 128,
                            .theta = 10000.0,
                            .scaling_type = RoPEScalingType::kNone};
    auto rope_or = AddRoPE(graph, 0U, q, k, position, params, "source_rope");
    ASSERT_TRUE(rope_or.ok()) << rope_or.status().ToString();
    graph.MarkOutput(rope_or->q);
    GraphRewriteSession session(graph);
    const GraphValueId virtual_k = session.AllocateVirtualValue();
    ReplacementNode prefix{
            .op_type = OpType::kRoPE,
            .decoder_layer_index = 0U,
            .inputs = {q, k, position},
            .outputs = {Replaces(rope_or->q, "live_q"),
                        {.desc = ActivationDesc("virtual_k"), .replaces = virtual_k}},
            .op_params = params,
            .name = "live_prefix"};
    ReplacementNode suffix =
            SiluReplacement(virtual_k, rope_or->k, "dead_suffix");
    const std::array old_nodes{ProducerOf(graph, rope_or->q)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes, {std::move(prefix), std::move(suffix)})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->FindNodesByOpType(OpType::kRoPE).size(), 1U);
    EXPECT_EQ(committed->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_TRUE(HasNodeNamed(*committed, "live_prefix"));
    EXPECT_FALSE(HasNodeNamed(*committed, "dead_suffix"));
}

TEST(CommitPruning, KeepsSideEffectfulReplacement) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId k_new = AddActivation(graph, "k_new");
    const GraphValueId v_new = AddActivation(graph, "v_new");
    const GraphValueId k_cache = AddState(
            graph,
            Spec(DataType::Float32(), {1, 2, 4}),
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
            "k_cache");
    const GraphValueId v_cache = AddState(
            graph,
            Spec(DataType::Float32(), {1, 2, 4}),
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue},
            "v_cache");
    auto cache_or = AddKVCacheUpdate(
            graph, 0U, k_new, v_new, k_cache, v_cache, "source_cache_update");
    ASSERT_TRUE(cache_or.ok()) << cache_or.status().ToString();
    graph.MarkOutput(live);
    GraphRewriteSession session(graph);
    const GraphNodeId cache_node = ProducerOf(graph, cache_or->k);
    const GraphNode& source = graph.GetNode(cache_node);
    ReplacementNode replacement{
            .op_type = OpType::kKVCacheUpdate,
            .decoder_layer_index = 0U,
            .inputs = source.inputs,
            .outputs = {{.desc = {.payload = graph.GetValue(cache_or->k).payload},
                         .replaces = cache_or->k},
                        {.desc = {.payload = graph.GetValue(cache_or->v).payload},
                         .replaces = cache_or->v}},
            .op_params = KVCacheUpdateParams{},
            .name = "retained_cache_update"};
    const std::array old_nodes{cache_node};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->FindNodesByOpType(OpType::kKVCacheUpdate).size(), 1U);
    EXPECT_TRUE(HasNodeNamed(*committed, "retained_cache_update"));
}

TEST(CommitPruning, KeepsSideEffectfulSourceNode) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId k_new = AddActivation(graph, "k_new");
    const GraphValueId v_new = AddActivation(graph, "v_new");
    const GraphValueId k_cache = AddState(
            graph,
            Spec(DataType::Float32(), {1, 2, 4}),
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
            "k_cache");
    const GraphValueId v_cache = AddState(
            graph,
            Spec(DataType::Float32(), {1, 2, 4}),
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue},
            "v_cache");
    auto cache_or = AddKVCacheUpdate(
            graph, 0U, k_new, v_new, k_cache, v_cache, "source_cache_update");
    ASSERT_TRUE(cache_or.ok()) << cache_or.status().ToString();
    const GraphValueId query = AddActivation(graph, "query");
    const AttentionParams attention_params{
            .num_attention_heads = 1,
            .num_key_value_heads = 1,
            .head_dim = 4};
    auto target_or = AddAttention(graph,
                                  0U,
                                  query,
                                  cache_or->k,
                                  cache_or->v,
                                  attention_params,
                                  "dead_target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    graph.MarkOutput(live);
    GraphRewriteSession session(graph);
    ReplacementNode replacement{
            .op_type = OpType::kAttention,
            .decoder_layer_index = 0U,
            .inputs = {query, cache_or->k, cache_or->v},
            .outputs = {Replaces(*target_or, "dead_attention.output")},
            .op_params = attention_params,
            .name = "dead_replacement"};
    const std::array old_nodes{ProducerOf(graph, *target_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(old_nodes, {std::move(replacement)}).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->FindNodesByOpType(OpType::kKVCacheUpdate).size(), 1U);
    EXPECT_EQ(committed->FindNodesByOpType(OpType::kAttention).size(), 0U);
    EXPECT_FALSE(HasNodeNamed(*committed, "dead_replacement"));
}

TEST(CommitPruning, MirrorRewriteNotPruned) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.RedirectInput(GraphNodeId{.index = 1},
                                      0,
                                      GraphValueId{.index = 0})
                        .ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 2U);
}

TEST(CommitPruning, PruneAfterConsumerRemoved) {
    ModelGraph graph;
    const GraphValueId live = AddActivation(graph, "live");
    const GraphValueId input = AddActivation(graph, "replacement_input");
    auto target_or = AddSilu(graph, 0U, input, "target");
    ASSERT_TRUE(target_or.ok()) << target_or.status().ToString();
    auto consumer_or = AddSilu(graph, 0U, *target_or, "removed_consumer");
    ASSERT_TRUE(consumer_or.ok()) << consumer_or.status().ToString();
    graph.MarkOutput(live);
    GraphRewriteSession session(graph);
    const std::array old_nodes{ProducerOf(graph, *target_or)};
    ASSERT_TRUE(session.ReplaceSubgraph(
                               old_nodes,
                               {SiluReplacement(input, *target_or, "orphan_after_remove")})
                        .ok());
    ASSERT_TRUE(session.RemoveNode(ProducerOf(graph, *consumer_or)).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_FALSE(HasNodeNamed(*committed, "orphan_after_remove"));
    EXPECT_FALSE(HasNodeNamed(*committed, "replacement_input"));
}

TEST(CommitPruning, ConstantPruningAligned) {
    const ModelGraph graph = BuildConstantRewriteGraph();
    GraphRewriteSession session(graph);
    InstallConstantRewritesPass pass;
    ASSERT_TRUE(pass.Run(session, {}).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_TRUE(HasValueNamed(*committed, "retained.constant"));
    EXPECT_FALSE(HasValueNamed(*committed, "pruned.constant"));
    EXPECT_TRUE(HasNodeNamed(*committed, "retained_replacement"));
    EXPECT_FALSE(HasNodeNamed(*committed, "pruned_replacement"));
}

TEST(CommitPruningGate, DefaultCommitDoesNotPrune) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), graph.GetNodes().size());
    EXPECT_EQ(committed->GetValues().size(), graph.GetValues().size());
}

TEST(CommitPruningGate, EmptyPipelineDoesNotPrune) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphPassManager pipeline;

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), graph.GetNodes().size());
}

TEST(CommitPruningGate, ExplicitCommitOptionEnablesPruning) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
}

TEST(CommitPruningGate, SessionRequestEnablesPruning) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    session.RequestCommitPruning();

    const StatusOr<ModelGraph> committed = session.Commit();

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
}

TEST(CommitPruningGate, RepeatedCommitAfterPruningRequestIsStable) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphRewriteSession session(graph);
    session.RequestCommitPruning();

    const StatusOr<ModelGraph> first = session.Commit();
    const StatusOr<ModelGraph> second = session.Commit();

    ASSERT_TRUE(first.ok()) << first.status().ToString();
    ASSERT_TRUE(second.ok()) << second.status().ToString();
    ExpectSameGraphStructure(*first, *second);
}

TEST(CommitPruningGate, DisabledDcePipelineDoesNotPrune) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    PassContext context;
    context.enable_dce = false;
    GraphPassManager pipeline(context);
    pipeline.Add(std::make_unique<DeadCodeEliminationPass>());

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), graph.GetNodes().size());
}

TEST(CommitPruningGate, NonDceNonEmptyPipelineDoesNotPrune) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<InstallOrphanReplacementPass>());

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 2U);
    EXPECT_TRUE(HasNodeNamed(*committed, "orphan_replacement"));
}

TEST(CommitPruningGate, NoopPipelineDoesNotPrune) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<NoopPass>());

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), graph.GetNodes().size());
}

TEST(CommitPruningGate, CheckpointBeforeDceDoesNotPruneEarly) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(1)
            .Add(std::make_unique<InstallOrphanReplacementPass>())
            .Add(std::make_unique<ExpectOrphanReplacementPass>())
            .Add(std::make_unique<DeadCodeEliminationPass>());

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_FALSE(HasNodeNamed(*committed, "orphan_replacement"));
}

TEST(CommitPruningGate, CheckpointAfterEnabledDcePrunes) {
    const ModelGraph graph = BuildTwoEmbeddingGraph();
    GraphPassManager pipeline;
    pipeline.SetCheckpointEvery(2)
            .Add(std::make_unique<InstallOrphanReplacementPass>())
            .Add(std::make_unique<DeadCodeEliminationPass>())
            .Add(std::make_unique<NoopPass>());

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_EQ(committed->GetNodes().size(), 1U);
    EXPECT_FALSE(HasNodeNamed(*committed, "orphan_replacement"));
}

TEST(CommitPruningGate, EnabledDcePipelinePrunes) {
    const ModelGraph graph = BuildConstantRewriteGraph();
    GraphPassManager pipeline;
    pipeline.Add(std::make_unique<InstallConstantRewritesPass>())
            .Add(std::make_unique<DeadCodeEliminationPass>());

    const StatusOr<ModelGraph> committed = pipeline.Run(graph);

    ASSERT_TRUE(committed.ok()) << committed.status().ToString();
    EXPECT_TRUE(HasValueNamed(*committed, "retained.constant"));
    EXPECT_FALSE(HasValueNamed(*committed, "pruned.constant"));
    EXPECT_TRUE(HasNodeNamed(*committed, "retained_replacement"));
    EXPECT_FALSE(HasNodeNamed(*committed, "pruned_replacement"));
}

TEST(CommitPruningFailure, RemovedProducerOfGraphOutput) {
    ModelGraph graph;
    const GraphValueId output = AddActivation(graph, "removed_output_producer");
    graph.MarkOutput(output);
    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.RemoveNode(ProducerOf(graph, output)).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_FALSE(committed.ok());
    EXPECT_EQ(committed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(committed.status().message().find("unavailable during commit"),
              std::string::npos);
}

TEST(CommitPruningFailure, RemovedProducerConsumedByRetainedNode) {
    ModelGraph graph;
    const GraphValueId removed = AddActivation(graph, "removed_producer");
    auto retained_or = AddSilu(graph, 0U, removed, "retained_consumer");
    ASSERT_TRUE(retained_or.ok()) << retained_or.status().ToString();
    graph.MarkOutput(*retained_or);
    GraphRewriteSession session(graph);
    ASSERT_TRUE(session.RemoveNode(ProducerOf(graph, removed)).ok());

    const StatusOr<ModelGraph> committed = session.Commit(ForcePruning());

    ASSERT_FALSE(committed.ok());
    EXPECT_EQ(committed.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(committed.status().message().find("unavailable during commit"),
              std::string::npos);
}

} // namespace
