#include "../test_graph_helpers.h"
#include "aethermind/model/graph/compilation/graph_lowering.h"

#include "aethermind/model/graph/graph_builder.h"
#include "aethermind/operators/rmsnorm_op.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

struct TestStorage : RawStorage {};

TensorSpec HiddenSpec() {
    return Spec(DataType::Float32(), {1, 8});
}

TensorSpec TokenSpec() {
    return Spec(DataType::Int(64), {1});
}

TensorSpec WeightSpec() {
    return Spec(DataType::Float32(), {8});
}

TensorSpec KVSpec() {
    return Spec(DataType::Float32(), {1, 2, 2});
}

StateBinding KStateBinding(uint32_t decoder_layer_index = 0U) {
    return KVCacheStateBinding{.decoder_layer_index = decoder_layer_index,
                               .slot = KVCacheSlot::kKey};
}

StateBinding VStateBinding(uint32_t decoder_layer_index = 0U) {
    return KVCacheStateBinding{.decoder_layer_index = decoder_layer_index,
                               .slot = KVCacheSlot::kValue};
}

GraphValueId AddActivation(ModelGraph& graph, TensorSpec spec, std::string name) {
    const GraphValueId tokens = graph.AddInput(TokenSpec(), name + "_tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.slot = ParameterSlot::kEmbeddingTable, .semantic_role = TransformerWeightRole::kTokenEmbedding});
    auto embed_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});
    AM_CHECK(embed_or.ok(), "AddActivation AddNode failed");
    return embed_or->outputs[0];
}

HfModelConfig MakeLlamaConfig(int64_t num_layers) {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 8,
            .intermediate_size = 16,
            .num_hidden_layers = num_layers,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .vocab_size = 32,
            .max_position_embeddings = 128,
            .head_dim = 2,
            .rms_norm_eps = 1.0e-5,
            .hidden_act = "silu",
            .tie_word_embeddings = false,
            .weight_dtype_hint = DataType::Float32(),
    };
}

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage,
                             std::vector<int64_t> shape) {
    return RawWeightView{
            .data = nullptr,
            .bytes = 0,
            .dtype = DataType::Float32(),
            .shape = std::move(shape),
            .storage = storage,
            .is_contiguous = true,
    };
}

ResolvedModelWeights MakeWeights(const HfModelConfig& config) {
    const auto storage = std::make_shared<TestStorage>();
    ResolvedModelWeights weights{
            .embed_tokens = MakeWeightView(storage, {config.vocab_size, config.hidden_size}),
            .final_norm = MakeWeightView(storage, {config.hidden_size}),
            .lm_head = MakeWeightView(storage, {config.vocab_size, config.hidden_size}),
    };

    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim : config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden_size = config.num_key_value_heads * head_dim;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        weights.layers.push_back(DecoderLayerRawWeights{
                .norm = NormRawWeights{
                        .input_rmsnorm = MakeWeightView(storage, {config.hidden_size}),
                        .post_attn_rmsnorm = MakeWeightView(storage, {config.hidden_size}),
                },
                .attn = AttnRawWeights{
                        .q_proj = MakeWeightView(storage, {config.hidden_size, config.hidden_size}),
                        .k_proj = MakeWeightView(storage, {kv_hidden_size, config.hidden_size}),
                        .v_proj = MakeWeightView(storage, {kv_hidden_size, config.hidden_size}),
                        .o_proj = MakeWeightView(storage, {config.hidden_size, config.hidden_size}),
                },
                .mlp = MLPRawWeights{
                        .gate_proj = MakeWeightView(storage, {config.intermediate_size, config.hidden_size}),
                        .up_proj = MakeWeightView(storage, {config.intermediate_size, config.hidden_size}),
                        .down_proj = MakeWeightView(storage, {config.hidden_size, config.intermediate_size}),
                },
        });
    }

    return weights;
}

TEST(GraphLowering, LowersEmbeddingGraphToExecutionStep) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.slot = ParameterSlot::kEmbeddingTable, .semantic_role = TransformerWeightRole::kTokenEmbedding});
    auto embedding_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});
    ASSERT_TRUE(embedding_or.ok()) << embedding_or.status().ToString();
    const AddedNode& embedding = *embedding_or;
    graph.MarkOutput(embedding.outputs[0], "hidden");

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 1U);
    EXPECT_EQ(lowered->steps[0].op_type, OpType::kEmbedding);
    EXPECT_EQ(lowered->steps[0].act_dtype, DataType::Float32());
    EXPECT_EQ(lowered->steps[0].weight_dtype, DataType::Float32());
    ASSERT_EQ(lowered->steps[0].input_specs.size(), 2U);
    EXPECT_EQ(lowered->steps[0].input_specs[0].dtype, DataType::Int(64));
    EXPECT_EQ(lowered->steps[0].input_specs[1].dtype, DataType::Float32());
    ASSERT_EQ(lowered->steps[0].output_specs.size(), 1U);
    EXPECT_EQ(lowered->steps[0].output_specs[0], HiddenSpec());
    ASSERT_EQ(lowered->step_bindings.size(), 1U);
    EXPECT_EQ(lowered->step_bindings[0].node, embedding.node);
    ASSERT_EQ(lowered->step_bindings[0].input_values.size(), 2U);
    EXPECT_EQ(lowered->step_bindings[0].input_values[0], tokens);
    EXPECT_EQ(lowered->step_bindings[0].input_values[1], weight);
    ASSERT_EQ(lowered->model_inputs.size(), 1U);
    EXPECT_EQ(lowered->model_inputs[0], tokens);
    ASSERT_EQ(lowered->model_outputs.size(), 1U);
    EXPECT_EQ(lowered->model_outputs[0], embedding.outputs[0]);
}

TEST(GraphLowering, PreservesTopologicalOrderAndRmsNormParams) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId embedding_weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.slot = ParameterSlot::kEmbeddingTable, .semantic_role = TransformerWeightRole::kTokenEmbedding});
    auto embedding_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, embedding_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});
    ASSERT_TRUE(embedding_or.ok()) << embedding_or.status().ToString();
    const AddedNode& embedding = *embedding_or;
    const GraphValueId norm_weight = graph.AddWeight(WeightSpec(), WeightBinding{.slot = ParameterSlot::kScale, .semantic_role = TransformerWeightRole::kFinalNorm});
    auto rms_norm_or = graph.AddNode(
            OpType::kRmsNorm,
            std::nullopt,
            {embedding.outputs[0], norm_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            RmsNormParams{.eps = 2.5e-3F});
    ASSERT_TRUE(rms_norm_or.ok()) << rms_norm_or.status().ToString();
    const AddedNode& rms_norm = *rms_norm_or;
    graph.MarkOutput(rms_norm.outputs[0], "normed");

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 2U);
    EXPECT_EQ(lowered->steps[0].op_type, OpType::kEmbedding);
    EXPECT_EQ(lowered->steps[1].op_type, OpType::kRmsNorm);
    const auto* params = std::get_if<RmsNormOp::Params>(&lowered->steps[1].op_params);
    ASSERT_NE(params, nullptr);
    EXPECT_FLOAT_EQ(params->eps, 2.5e-3F);
    ASSERT_EQ(lowered->step_bindings[1].input_values.size(), 2U);
    EXPECT_EQ(lowered->step_bindings[1].input_values[0], embedding.outputs[0]);
    EXPECT_EQ(lowered->step_bindings[1].input_values[1], norm_weight);
    ASSERT_EQ(lowered->steps[1].output_specs.size(), 1U);
    EXPECT_EQ(lowered->steps[1].output_specs[0], HiddenSpec());
}

TEST(GraphLowering, RecordsKVCacheUpdateLoweringTimeStateAliases) {
    ModelGraph graph;
    const GraphValueId k = AddActivation(graph, KVSpec(), "k");
    const GraphValueId v = AddActivation(graph, KVSpec(), "v");
    const GraphValueId k_state_in = graph.AddState(KVSpec(), KStateBinding(), "k_cache_in");
    const GraphValueId v_state_in = graph.AddState(KVSpec(), VStateBinding(), "v_cache_in");
    auto update_or = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_state_in, v_state_in},
            {NodeOutputDesc{.payload = StateValue{.binding = KStateBinding()}},
             NodeOutputDesc{.payload = StateValue{.binding = VStateBinding()}}},
            KVCacheUpdateParams{});
    ASSERT_TRUE(update_or.ok()) << update_or.status().ToString();
    const AddedNode& update = *update_or;

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 3U);
    EXPECT_EQ(lowered->steps[2].op_type, OpType::kKVCacheUpdate);
    ASSERT_EQ(lowered->steps[2].input_specs.size(), 2U);
    ASSERT_EQ(lowered->step_bindings[2].input_values.size(), 4U);
    EXPECT_EQ(lowered->step_bindings[2].input_values[2], k_state_in);
    EXPECT_EQ(lowered->step_bindings[2].input_values[3], v_state_in);
    ASSERT_EQ(lowered->steps[2].output_specs.size(), 2U);
    EXPECT_EQ(lowered->steps[2].output_specs[0], KVSpec());
    EXPECT_EQ(lowered->steps[2].output_specs[1], KVSpec());
    ASSERT_EQ(lowered->state_aliases.size(), 2U);
    EXPECT_EQ(lowered->state_aliases[0].input, k_state_in);
    EXPECT_EQ(lowered->state_aliases[0].output, update.outputs[0]);
    EXPECT_EQ(lowered->state_aliases[1].input, v_state_in);
    EXPECT_EQ(lowered->state_aliases[1].output, update.outputs[1]);
}

TEST(GraphLowering, LowersAttentionStatePortsWithoutTensorSpecs) {
    ModelGraph graph;
    const GraphValueId q = AddActivation(graph, HiddenSpec(), "q");
    const GraphValueId k_cache = graph.AddState(KVSpec(), KStateBinding(), "k_cache");
    const GraphValueId v_cache = graph.AddState(KVSpec(), VStateBinding(), "v_cache");
    auto attention_or = graph.AddNode(
            OpType::kAttention,
            0U,
            {q, k_cache, v_cache},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            AttentionParams{.num_attention_heads = 4, .num_key_value_heads = 2, .head_dim = 2});
    ASSERT_TRUE(attention_or.ok()) << attention_or.status().ToString();
    const AddedNode& attention = *attention_or;
    graph.MarkOutput(attention.outputs[0], "attention_output");

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 2U);
    EXPECT_EQ(lowered->steps[1].op_type, OpType::kAttention);
    ASSERT_EQ(lowered->steps[1].input_specs.size(), 1U);
    EXPECT_EQ(lowered->steps[1].input_specs[0].dtype, DataType::Float32());
    ASSERT_EQ(lowered->steps[1].output_specs.size(), 1U);
    EXPECT_EQ(lowered->steps[1].output_specs[0], HiddenSpec());
    ASSERT_EQ(lowered->step_bindings[1].input_values.size(), 3U);
    EXPECT_EQ(lowered->step_bindings[1].input_values[1], k_cache);
    EXPECT_EQ(lowered->step_bindings[1].input_values[2], v_cache);
}

TEST(GraphLowering, RejectsInvalidGraph) {
    const ModelGraph graph({}, {}, {GraphValue{}});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_FALSE(lowered.ok());
    EXPECT_EQ(lowered.status().code(), StatusCode::kInvalidArgument);
}

TEST(GraphLowering, LowersFullLlamaDenseGraph) {
    const HfModelConfig config = MakeLlamaConfig(2);
    const ResolvedModelWeights weights = MakeWeights(config);
    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);
    ASSERT_TRUE(graph.ok()) << graph.status().ToString();

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(*graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), graph->GetNodes().size());
    ASSERT_EQ(lowered->step_bindings.size(), graph->GetNodes().size());
    EXPECT_EQ(lowered->steps.front().op_type, OpType::kEmbedding);
    EXPECT_EQ(lowered->steps.back().op_type, OpType::kArgmax);
    EXPECT_EQ(lowered->model_inputs.size(), graph->GetInputs().size());
    EXPECT_EQ(lowered->model_outputs.size(), graph->GetOutputs().size());
    EXPECT_EQ(lowered->state_aliases.size(), static_cast<size_t>(config.num_hidden_layers) * 2U);
    for (size_t i = 0; i < lowered->steps.size(); ++i) {
        EXPECT_EQ(lowered->steps[i].output_specs.size(), lowered->step_bindings[i].output_values.size());
    }
}

TEST(GraphLowering, ResolveStateAliasesConvertsLoweringTimeRecordsToRuntimePlan) {
    ModelGraph graph;
    const GraphValueId k = AddActivation(graph, KVSpec(), "k");
    const GraphValueId v = AddActivation(graph, KVSpec(), "v");
    const GraphValueId k_state_in = graph.AddState(KVSpec(), KStateBinding(), "k_cache_in");
    const GraphValueId v_state_in = graph.AddState(KVSpec(), VStateBinding(), "v_cache_in");
    (void) graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_state_in, v_state_in},
            {NodeOutputDesc{.payload = StateValue{.binding = KStateBinding()}},
             NodeOutputDesc{.payload = StateValue{.binding = VStateBinding()}}},
            KVCacheUpdateParams{});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok());

    const StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(*lowered);
    ASSERT_TRUE(alias_plan.ok()) << alias_plan.status().ToString();
    EXPECT_FALSE(alias_plan->empty());
    ASSERT_EQ(alias_plan->size(), 2U);

    // The KVCacheUpdate step is the 3rd step (index 2) because k and v
    // are created via AddActivation which adds an Embedding step each.
    const auto kvcache_aliases = alias_plan->ForStep(2);
    ASSERT_EQ(kvcache_aliases.size(), 2U);

    // K cache alias: port 2 in (k_state_in) → port 0 out (k_cache_out)
    EXPECT_EQ(kvcache_aliases[0].input_port, 2U);
    EXPECT_EQ(kvcache_aliases[0].output_port, 0U);
    // V cache alias: port 3 in (v_state_in) → port 1 out (v_cache_out)
    EXPECT_EQ(kvcache_aliases[1].input_port, 3U);
    EXPECT_EQ(kvcache_aliases[1].output_port, 1U);
}

TEST(GraphLowering, ResolveStateAliasesReturnsEmptyRuntimePlanForGraphWithoutState) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable, .semantic_role = TransformerWeightRole::kTokenEmbedding});
    (void) graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok());

    const StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(*lowered);
    ASSERT_TRUE(alias_plan.ok());
    EXPECT_TRUE(alias_plan->empty());
    EXPECT_EQ(alias_plan->size(), 0U);
}

TEST(GraphLowering, StateAliasPlanForStepReturnsEmptySpanForUnknownStep) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable, .semantic_role = TransformerWeightRole::kTokenEmbedding});
    (void) graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok());

    const StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(*lowered);
    ASSERT_TRUE(alias_plan.ok());

    const auto aliases = alias_plan->ForStep(999);
    EXPECT_TRUE(aliases.empty());
}

TEST(GraphLowering, ResolveStateAliasesFailsOnOrphanAlias) {
    ModelGraph graph;
    const GraphValueId k = AddActivation(graph, KVSpec(), "k");
    const GraphValueId v = AddActivation(graph, KVSpec(), "v");
    const GraphValueId k_state_in = graph.AddState(KVSpec(), KStateBinding(), "k_cache_in");
    const GraphValueId v_state_in = graph.AddState(KVSpec(), VStateBinding(), "v_cache_in");
    (void) graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_state_in, v_state_in},
            {NodeOutputDesc{.payload = StateValue{.binding = KStateBinding()}},
             NodeOutputDesc{.payload = StateValue{.binding = VStateBinding()}}},
            KVCacheUpdateParams{});

    StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);
    ASSERT_TRUE(lowered.ok());

    // Inject a bogus alias referencing a non-existent GraphValueId.
    lowered->state_aliases.push_back(LoweredStateAlias{
            .input = GraphValueId{.index = 99999},
            .output = k_state_in,
    });

    const StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(*lowered);
    ASSERT_FALSE(alias_plan.ok());
    EXPECT_EQ(alias_plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(GraphLowering, WeightlessOpFallsBackWeightDTypeToActDType) {
    ModelGraph graph;
    const GraphValueId lhs = AddActivation(graph, HiddenSpec(), "lhs");
    const GraphValueId rhs = AddActivation(graph, HiddenSpec(), "rhs");
    (void) graph.AddNode(
            OpType::kAdd,
            std::nullopt,
            {lhs, rhs},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            AddParams{});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 3U);
    const auto& add_step = lowered->steps[2];
    EXPECT_EQ(add_step.op_type, OpType::kAdd);
    EXPECT_EQ(add_step.act_dtype, DataType::Float32());
    EXPECT_EQ(add_step.weight_dtype, DataType::Float32());
}

TEST(GraphLowering, WeightedOpPreservesOriginalWeightDType) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    (void) graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            EmbeddingParams{});

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(graph);

    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), 1U);
    EXPECT_EQ(lowered->steps[0].op_type, OpType::kEmbedding);
    EXPECT_EQ(lowered->steps[0].weight_dtype, DataType::Float32());
}

}// namespace
}// namespace aethermind
