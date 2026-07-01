#include "aethermind/model/graph/graph_op_builder.h"

#include <gtest/gtest.h>

#include <optional>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return TensorSpec{.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

TEST(GraphOpBuilder, AddsSingleOutputOperatorHelpers) {
    ModelGraph graph;
    const TensorSpec token_spec = Spec(DataType::Int(64), {2});
    const TensorSpec hidden_spec = Spec(DataType::Float32(), {2, 4});
    const TensorSpec logits_spec = Spec(DataType::Float32(), {2, 16});
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 8, 2});
    const GraphValueId tokens = graph.AddInput(token_spec, "tokens");
    const GraphValueId k_cache = graph.AddState(cache_spec,
                                                KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
                                                "k_cache");
    const GraphValueId v_cache = graph.AddState(cache_spec,
                                                KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue},
                                                "v_cache");

    const GraphValueId hidden = AddEmbedding(graph,
                                             tokens,
                                             16,
                                             4,
                                             DataType::Float32(),
                                             WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                           .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                             "embedding");
    const GraphValueId normed = AddRmsNorm(graph,
                                           hidden,
                                           DataType::Float32(),
                                           WeightBinding{.slot = ParameterSlot::kScale,
                                                         .decoder_layer_index = 0,
                                                         .semantic_role = TransformerWeightRole::kInputNorm},
                                           1.0e-5F,
                                           "norm");
    const GraphValueId q = AddLinear(graph,
                                     normed,
                                     4,
                                     DataType::Float32(),
                                     WeightBinding{.slot = ParameterSlot::kKernel,
                                                   .decoder_layer_index = 0,
                                                   .semantic_role = TransformerWeightRole::kAttentionQ},
                                     "q_proj");
    const GraphValueId attn = AddAttention(graph,
                                           0,
                                           q,
                                           k_cache,
                                           v_cache,
                                           AttentionParams{.num_attention_heads = 2,
                                                           .num_key_value_heads = 2,
                                                           .head_dim = 2},
                                           "attention");
    EXPECT_EQ(graph.GetValue(attn).spec, graph.GetValue(q).spec);
    const GraphValueId sum = AddElementwiseAdd(graph, 0, hidden, attn, "add");
    EXPECT_EQ(graph.GetValue(sum).spec, graph.GetValue(hidden).spec);
    const GraphValueId act = AddSiluMul(graph, 0, sum, sum, "silu_mul");
    EXPECT_EQ(graph.GetValue(act).spec, graph.GetValue(sum).spec);
    const GraphValueId logits = AddLinear(graph,
                                          act,
                                          16,
                                          DataType::Float32(),
                                          WeightBinding{.slot = ParameterSlot::kKernel,
                                                        .semantic_role = TransformerWeightRole::kLmHead},
                                          "lm_head");
    const GraphValueId output = AddArgmax(graph, std::nullopt, logits, token_spec, -1, "argmax");
    graph.MarkOutput(output, "output");

    const Status validation = graph.Validate();
    ASSERT_TRUE(validation.ok()) << validation.ToString();

    ASSERT_TRUE(graph.GetValue(hidden).producer.has_value());
    const GraphNode& embedding_node = graph.GetNode(*graph.GetValue(hidden).producer);
    ASSERT_EQ(embedding_node.inputs.size(), 2U);
    const GraphValue& embedding_weight = graph.GetValue(embedding_node.inputs[1]);
    EXPECT_EQ(embedding_weight.spec, Spec(DataType::Float32(), {16, 4}));
    EXPECT_EQ(embedding_weight.debug_name, "embedding");
    ASSERT_TRUE(std::holds_alternative<WeightValue>(embedding_weight.payload));
    const WeightBinding& embedding_binding = std::get<WeightValue>(embedding_weight.payload).binding;
    EXPECT_EQ(embedding_binding.slot, ParameterSlot::kEmbeddingTable);
    EXPECT_FALSE(embedding_binding.decoder_layer_index.has_value());
    ASSERT_TRUE(std::holds_alternative<TransformerWeightRole>(embedding_binding.semantic_role));
    EXPECT_EQ(std::get<TransformerWeightRole>(embedding_binding.semantic_role), TransformerWeightRole::kTokenEmbedding);

    ASSERT_TRUE(graph.GetValue(normed).producer.has_value());
    const GraphNode& norm_node = graph.GetNode(*graph.GetValue(normed).producer);
    ASSERT_EQ(norm_node.inputs.size(), 2U);
    const GraphValue& norm_weight = graph.GetValue(norm_node.inputs[1]);
    EXPECT_EQ(norm_weight.spec, Spec(DataType::Float32(), {4}));
    EXPECT_EQ(norm_weight.debug_name, "norm");
    ASSERT_TRUE(std::holds_alternative<WeightValue>(norm_weight.payload));
    const WeightBinding& norm_binding = std::get<WeightValue>(norm_weight.payload).binding;
    EXPECT_EQ(norm_binding.slot, ParameterSlot::kScale);
    EXPECT_EQ(norm_binding.decoder_layer_index, std::optional<uint32_t>{0});
    ASSERT_TRUE(std::holds_alternative<TransformerWeightRole>(norm_binding.semantic_role));
    EXPECT_EQ(std::get<TransformerWeightRole>(norm_binding.semantic_role), TransformerWeightRole::kInputNorm);

    ASSERT_TRUE(graph.GetValue(q).producer.has_value());
    const GraphNode& q_node = graph.GetNode(*graph.GetValue(q).producer);
    ASSERT_EQ(q_node.inputs.size(), 2U);
    const GraphValue& q_weight = graph.GetValue(q_node.inputs[1]);
    EXPECT_EQ(q_weight.spec, Spec(DataType::Float32(), {4, 4}));
    EXPECT_EQ(q_weight.debug_name, "q_proj");
    ASSERT_TRUE(std::holds_alternative<WeightValue>(q_weight.payload));
    const WeightBinding& q_binding = std::get<WeightValue>(q_weight.payload).binding;
    EXPECT_EQ(q_binding.slot, ParameterSlot::kKernel);
    EXPECT_EQ(q_binding.decoder_layer_index, std::optional<uint32_t>{0});
    ASSERT_TRUE(std::holds_alternative<TransformerWeightRole>(q_binding.semantic_role));
    EXPECT_EQ(std::get<TransformerWeightRole>(q_binding.semantic_role), TransformerWeightRole::kAttentionQ);

    ASSERT_TRUE(graph.GetValue(logits).producer.has_value());
    const GraphNode& logits_node = graph.GetNode(*graph.GetValue(logits).producer);
    ASSERT_EQ(logits_node.inputs.size(), 2U);
    const GraphValue& lm_head_weight = graph.GetValue(logits_node.inputs[1]);
    EXPECT_EQ(lm_head_weight.spec, Spec(DataType::Float32(), {16, 4}));
    EXPECT_EQ(lm_head_weight.debug_name, "lm_head");
    ASSERT_TRUE(std::holds_alternative<WeightValue>(lm_head_weight.payload));
    const WeightBinding& lm_head_binding = std::get<WeightValue>(lm_head_weight.payload).binding;
    EXPECT_EQ(lm_head_binding.slot, ParameterSlot::kKernel);
    EXPECT_FALSE(lm_head_binding.decoder_layer_index.has_value());
    ASSERT_TRUE(std::holds_alternative<TransformerWeightRole>(lm_head_binding.semantic_role));
    EXPECT_EQ(std::get<TransformerWeightRole>(lm_head_binding.semantic_role), TransformerWeightRole::kLmHead);

    EXPECT_EQ(graph.GetNode(GraphNodeId{.index = 0}).op_type, OpType::kEmbedding);
    EXPECT_EQ(graph.GetNode(GraphNodeId{.index = 5}).op_type, OpType::kSiluMul);
    EXPECT_TRUE(std::holds_alternative<ArgmaxParams>(graph.GetNode(GraphNodeId{.index = 7}).op_params));
}

TEST(GraphOpBuilder, AddSiluMulRequiresMatchingSpecs) {
    ModelGraph graph;
    const GraphValueId gate = graph.AddInput(Spec(DataType::Float32(), {2, 4}), "gate");
    const GraphValueId up = graph.AddInput(Spec(DataType::Float32(), {2, 8}), "up");

    EXPECT_DEATH(static_cast<void>(AddSiluMul(graph, 0, gate, up, "bad_silu_mul")),
                 "SiluMul gate and up specs must match");
}

TEST(GraphOpBuilder, AddLinearDerivesSpecsForRankOneInput) {
    ModelGraph graph(HfModelConfig{}, {}, {GraphValue{.payload = ActivationValue{}, .spec = Spec(DataType::Float32(), {4}), .debug_name = "input"}});
    const GraphValueId input{.index = 0};

    const GraphValueId output = AddLinear(graph,
                                          input,
                                          8,
                                          DataType::Float32(),
                                          WeightBinding{.slot = ParameterSlot::kKernel,
                                                        .semantic_role = TransformerWeightRole::kLmHead},
                                          "linear");

    EXPECT_EQ(graph.GetValue(output).spec, Spec(DataType::Float32(), {8}));

    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& linear_node = graph.GetNode(*graph.GetValue(output).producer);
    ASSERT_EQ(linear_node.inputs.size(), 2U);
    const GraphValue& weight = graph.GetValue(linear_node.inputs[1]);
    EXPECT_EQ(weight.spec, Spec(DataType::Float32(), {8, 4}));
    EXPECT_EQ(weight.debug_name, "linear");
}

TEST(GraphOpBuilder, AddLinearDerivesSpecsForHigherRankInput) {
    ModelGraph graph(HfModelConfig{}, {}, {GraphValue{.payload = ActivationValue{}, .spec = Spec(DataType::Float32(), {2, 3, 4}), .debug_name = "input"}});
    const GraphValueId input{.index = 0};

    const GraphValueId output = AddLinear(graph,
                                          input,
                                          8,
                                          DataType::Float32(),
                                          WeightBinding{.slot = ParameterSlot::kKernel,
                                                        .semantic_role = TransformerWeightRole::kLmHead},
                                          "linear");

    EXPECT_EQ(graph.GetValue(output).spec, Spec(DataType::Float32(), {2, 3, 8}));

    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& linear_node = graph.GetNode(*graph.GetValue(output).producer);
    ASSERT_EQ(linear_node.inputs.size(), 2U);
    const GraphValue& weight = graph.GetValue(linear_node.inputs[1]);
    EXPECT_EQ(weight.spec, Spec(DataType::Float32(), {8, 4}));
    EXPECT_EQ(weight.debug_name, "linear");
}

TEST(GraphOpBuilder, AddRmsNormDerivesSpecsForHigherRankInput) {
    ModelGraph graph(HfModelConfig{}, {}, {GraphValue{.payload = ActivationValue{}, .spec = Spec(DataType::Float32(), {2, 3, 4}), .debug_name = "input"}});
    const GraphValueId input{.index = 0};

    const GraphValueId output = AddRmsNorm(graph,
                                           input,
                                           DataType::Float32(),
                                           WeightBinding{.slot = ParameterSlot::kScale,
                                                         .semantic_role = TransformerWeightRole::kFinalNorm},
                                           1.0e-5F,
                                           "norm");

    EXPECT_EQ(graph.GetValue(output).spec, Spec(DataType::Float32(), {2, 3, 4}));

    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& norm_node = graph.GetNode(*graph.GetValue(output).producer);
    ASSERT_EQ(norm_node.inputs.size(), 2U);
    const GraphValue& weight = graph.GetValue(norm_node.inputs[1]);
    EXPECT_EQ(weight.spec, Spec(DataType::Float32(), {4}));
    EXPECT_EQ(weight.debug_name, "norm");
}

TEST(GraphOpBuilder, AddsMultiOutputOperatorHelpers) {
    ModelGraph graph;
    const TensorSpec hidden_spec = Spec(DataType::Float32(), {2, 4});
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 8, 2});
    const GraphValueId q = graph.AddInput(hidden_spec, "q");
    const GraphValueId k = graph.AddInput(hidden_spec, "k");
    const GraphValueId v = graph.AddInput(hidden_spec, "v");
    const GraphValueId position_ids = graph.AddInput(Spec(DataType::Int(64), {2}), "position_ids");
    const StateBinding k_binding = KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey};
    const StateBinding v_binding = KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue};
    const GraphValueId k_cache = graph.AddState(cache_spec, k_binding, "k_cache");
    const GraphValueId v_cache = graph.AddState(cache_spec, v_binding, "v_cache");

    const RoPEOutputs rope = AddRoPE(graph,
                                     0,
                                     q,
                                     k,
                                     position_ids,
                                     RoPEParams{.head_dim = 2,
                                                .num_attention_heads = 2,
                                                .num_key_value_heads = 2,
                                                .max_position_embeddings = 128},
                                     "rope");
    const KVCachePair cache = AddKVCacheUpdate(graph,
                                               0,
                                               rope.k,
                                               v,
                                               k_cache,
                                               v_cache,
                                               "kv_cache_update");

    EXPECT_EQ(graph.GetNode(GraphNodeId{.index = 0}).outputs.size(), 2U);
    EXPECT_EQ(graph.GetNode(GraphNodeId{.index = 1}).outputs.size(), 2U);
    EXPECT_TRUE(std::holds_alternative<ActivationValue>(graph.GetValue(rope.q).payload));
    EXPECT_TRUE(std::holds_alternative<StateValue>(graph.GetValue(cache.k).payload));
    EXPECT_TRUE(std::holds_alternative<StateValue>(graph.GetValue(cache.v).payload));
    EXPECT_EQ(graph.GetValue(cache.k).spec, cache_spec);
    EXPECT_EQ(graph.GetValue(cache.v).spec, cache_spec);
    const auto& k_state = std::get<StateValue>(graph.GetValue(cache.k).payload);
    const auto& v_state = std::get<StateValue>(graph.GetValue(cache.v).payload);
    EXPECT_EQ(std::get<KVCacheStateBinding>(k_state.binding).slot, KVCacheSlot::kKey);
    EXPECT_EQ(std::get<KVCacheStateBinding>(v_state.binding).slot, KVCacheSlot::kValue);
}

TEST(GraphOpBuilder, KVCacheUpdateRequiresStateCacheInputs) {
    ModelGraph graph;
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 4, 2});
    const GraphValueId k_new = graph.AddInput(cache_spec, "k_new");
    const GraphValueId v_new = graph.AddInput(cache_spec, "v_new");
    const GraphValueId activation_cache = graph.AddInput(cache_spec, "activation_cache");
    const GraphValueId k_cache = graph.AddState(cache_spec,
                                                KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
                                                "k_cache");
    const GraphValueId v_cache = graph.AddState(cache_spec,
                                                KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue},
                                                "v_cache");

    EXPECT_DEATH(static_cast<void>(AddKVCacheUpdate(graph,
                                                    0,
                                                    k_new,
                                                    v_new,
                                                    activation_cache,
                                                    v_cache,
                                                    "bad_k_cache")),
                 "K cache input must be a StateValue");
    EXPECT_DEATH(static_cast<void>(AddKVCacheUpdate(graph,
                                                    0,
                                                    k_new,
                                                    v_new,
                                                    k_cache,
                                                    activation_cache,
                                                    "bad_v_cache")),
                 "V cache input must be a StateValue");
}

TEST(GraphOpBuilder, AddInputAndAddStateRegisterExternalValues) {
    ModelGraph graph;
    const TensorSpec input_spec = Spec(DataType::Int(64), {4});
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 4, 2});

    const GraphValueId tokens = AddInput(graph, input_spec, "tokens");
    const StateBinding k_binding = KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey};
    const GraphValueId k_cache = AddState(graph, cache_spec, k_binding, "k_cache");

    ASSERT_EQ(graph.GetInputs().size(), 1U);
    EXPECT_EQ(graph.GetInputs().front().name, "tokens");
    EXPECT_EQ(graph.GetInputs().front().value, tokens);
    EXPECT_TRUE(std::holds_alternative<StateValue>(graph.GetValue(k_cache).payload));
    const auto& state = std::get<StateValue>(graph.GetValue(k_cache).payload);
    const auto& kv_binding = std::get<KVCacheStateBinding>(state.binding);
    EXPECT_EQ(kv_binding.decoder_layer_index, 0U);
    EXPECT_EQ(kv_binding.slot, KVCacheSlot::kKey);
}

}// namespace
}// namespace aethermind
