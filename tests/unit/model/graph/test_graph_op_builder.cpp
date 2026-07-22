#include "aethermind/model/graph/graph_op_builder.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "test_graph_helpers.h"

#include <gtest/gtest.h>
#include <optional>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

TEST(GraphOpBuilder, AddsSingleOutputOperatorHelpers) {
    ModelGraph graph;
    const TensorSpec token_spec = Spec(DataType::Int(64), {2});
    const TensorSpec hidden_spec = Spec(DataType::Float32(), {2, 4});
    const TensorSpec logits_spec = Spec(DataType::Float32(), {2, 16});
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 8, 2});
    const GraphValueId tokens = graph.AddInput(token_spec, "tokens");
    const GraphValueId k_cache = graph.AddState(
            cache_spec,
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
            "k_cache");
    const GraphValueId v_cache = graph.AddState(
            cache_spec,
            KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue},
            "v_cache");

    auto hidden_or = AddEmbedding(
            graph,
            tokens,
            16,
            4,
            DataType::Float32(),
            {.slot = ParameterSlot::kEmbeddingTable,
             .semantic_role = TransformerWeightRole::kTokenEmbedding},
            "embedding");
    ASSERT_TRUE(hidden_or.ok()) << hidden_or.status().ToString();
    const GraphValueId hidden = *hidden_or;
    auto normed_or = AddRmsNorm(
            graph,
            hidden,
            DataType::Float32(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0,
             .semantic_role = TransformerWeightRole::kInputNorm},
            1.0e-5F,
            "norm");
    ASSERT_TRUE(normed_or.ok()) << normed_or.status().ToString();
    const GraphValueId normed = *normed_or;
    auto q_or = AddLinear(
            graph,
            normed,
            4,
            DataType::Float32(),
            {.slot = ParameterSlot::kKernel,
             .decoder_layer_index = 0,
             .semantic_role = TransformerWeightRole::kAttentionQ},
            "q_proj");
    ASSERT_TRUE(q_or.ok()) << q_or.status().ToString();
    const GraphValueId q = *q_or;
    auto attn_or = AddAttention(
            graph,
            0,
            q,
            k_cache,
            v_cache,
            AttentionParams{.num_attention_heads = 2,
                            .num_key_value_heads = 2,
                            .head_dim = 2},
            "attention");
    ASSERT_TRUE(attn_or.ok()) << attn_or.status().ToString();
    const GraphValueId attn = *attn_or;
    EXPECT_EQ(graph.GetValue(attn).spec, graph.GetValue(q).spec);
    auto sum_or = AddElementwiseAdd(graph, 0, hidden, attn, "add");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    EXPECT_EQ(graph.GetValue(sum).spec, graph.GetValue(hidden).spec);
    auto act_or = AddSiluMul(graph, 0, sum, sum, "silu_mul");
    ASSERT_TRUE(act_or.ok()) << act_or.status().ToString();
    const GraphValueId act = *act_or;
    EXPECT_EQ(graph.GetValue(act).spec, graph.GetValue(sum).spec);
    auto logits_or = AddLinear(
            graph,
            act,
            16,
            DataType::Float32(),
            {.slot = ParameterSlot::kKernel,
             .semantic_role = TransformerWeightRole::kLmHead},
            "lm_head");
    ASSERT_TRUE(logits_or.ok()) << logits_or.status().ToString();
    const GraphValueId logits = *logits_or;
    auto output_or = AddArgmax(graph, std::nullopt, logits, -1, "argmax");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;
    graph.MarkOutput(output);

    const Status validation = graph.Validate();
    ASSERT_TRUE(validation.ok()) << validation.ToString();

    ASSERT_TRUE(graph.GetValue(hidden).producer.has_value());
    const GraphNode& embedding_node = graph.GetNode(*graph.GetValue(hidden).producer);
    ASSERT_EQ(embedding_node.inputs.size(), 2U);
    const GraphValue& embedding_weight = graph.GetValue(embedding_node.inputs[1]);
    EXPECT_EQ(embedding_weight.spec, Spec(DataType::Float32(), {16, 4}));
    EXPECT_EQ(embedding_weight.name, "embedding");
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
    EXPECT_EQ(norm_weight.name, "norm");
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
    EXPECT_EQ(q_weight.name, "q_proj");
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
    EXPECT_EQ(lm_head_weight.name, "lm_head");
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
    const GraphValueId gate = graph.AddConstant(
            Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "gate");
    const GraphValueId up = graph.AddConstant(
            Spec(DataType::Float32(), {2, 8}), ConstantBinding{}, "up");

    EXPECT_FALSE(AddSiluMul(graph, 0, gate, up, "bad_silu_mul").ok());
}

TEST(GraphOpBuilder, AddSiluDerivesOutputSpecFromInput) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "input");

    auto output_or = AddSilu(graph, 0, input, "silu");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, graph.GetValue(input).spec);
    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& node = graph.GetNode(*graph.GetValue(output).producer);
    EXPECT_EQ(node.op_type, OpType::kSilu);
    EXPECT_EQ(node.inputs.size(), 1U);
    EXPECT_EQ(node.outputs.size(), 1U);
    EXPECT_TRUE(std::holds_alternative<SiluParams>(node.op_params));
}

TEST(GraphOpBuilder, AddElementwiseMulDerivesOutputSpecFromLhs) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(
            Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(
            Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseMul(graph, 0, lhs, rhs, "mul");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, graph.GetValue(lhs).spec);
    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& node = graph.GetNode(*graph.GetValue(output).producer);
    EXPECT_EQ(node.op_type, OpType::kElementwiseMul);
    EXPECT_EQ(node.inputs.size(), 2U);
    EXPECT_EQ(node.outputs.size(), 1U);
    EXPECT_TRUE(std::holds_alternative<ElementwiseMulParams>(node.op_params));
}

TEST(GraphOpBuilder, AddElementwiseMulRequiresMatchingSpecs) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {2, 8}), ConstantBinding{}, "rhs");

    EXPECT_FALSE(AddElementwiseMul(graph, 0, lhs, rhs, "bad_mul").ok());
}

TEST(GraphOpBuilder, AddLinearDerivesSpecsForRankOneInput) {
    ModelGraph graph(HfModelConfig{}, {}, {GraphValue{.payload = ActivationValue{}, .spec = Spec(DataType::Float32(), {4}), .name = "input"}});
    const GraphValueId input{.index = 0};

    auto output_or = AddLinear(graph,
                               input,
                               8,
                               DataType::Float32(),
                               WeightBinding{.slot = ParameterSlot::kKernel,
                                             .semantic_role = TransformerWeightRole::kLmHead},
                               "linear");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, Spec(DataType::Float32(), {8}));

    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& linear_node = graph.GetNode(*graph.GetValue(output).producer);
    ASSERT_EQ(linear_node.inputs.size(), 2U);
    const GraphValue& weight = graph.GetValue(linear_node.inputs[1]);
    EXPECT_EQ(weight.spec, Spec(DataType::Float32(), {8, 4}));
    EXPECT_EQ(weight.name, "linear");
}

TEST(GraphOpBuilder, AddLinearRejectsRankThreeInput) {
    ModelGraph graph(HfModelConfig{}, {}, {GraphValue{.payload = ActivationValue{}, .spec = Spec(DataType::Float32(), {2, 3, 4}), .name = "input"}});
    const GraphValueId input{.index = 0};
    EXPECT_FALSE(AddLinear(graph,
                           input,
                           8,
                           DataType::Float32(),
                           WeightBinding{.slot = ParameterSlot::kKernel,
                                         .semantic_role = TransformerWeightRole::kLmHead},
                           "linear")
                         .ok());
}

TEST(GraphOpBuilder, AddRmsNormRejectsRankThreeInput) {
    ModelGraph graph(HfModelConfig{}, {}, {GraphValue{.payload = ActivationValue{}, .spec = Spec(DataType::Float32(), {2, 3, 4}), .name = "input"}});
    const GraphValueId input{.index = 0};
    EXPECT_FALSE(AddRmsNorm(graph,
                            input,
                            DataType::Float32(),
                            WeightBinding{.slot = ParameterSlot::kScale,
                                          .semantic_role = TransformerWeightRole::kFinalNorm},
                            1.0e-5F,
                            "norm")
                         .ok());
}

TEST(GraphOpBuilder, AddsMultiOutputOperatorHelpers) {
    ModelGraph graph;
    const TensorSpec hidden_spec = Spec(DataType::Float32(), {2, 4});
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 8, 2});
    const GraphValueId q = graph.AddConstant(hidden_spec, ConstantBinding{}, "q");
    const GraphValueId k = graph.AddConstant(hidden_spec, ConstantBinding{}, "k");
    const GraphValueId position_ids = graph.AddInput(Spec(DataType::Int(64), {2}), "position_ids");
    const StateBinding k_binding = KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey};
    const StateBinding v_binding = KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue};
    const GraphValueId k_cache = graph.AddState(cache_spec, k_binding, "k_cache");
    const GraphValueId v_cache = graph.AddState(cache_spec, v_binding, "v_cache");
    const GraphValueId v = graph.AddConstant(hidden_spec, ConstantBinding{}, "v");

    auto rope_or = AddRoPE(graph,
                           0,
                           q,
                           k,
                           position_ids,
                           RoPEParams{.head_dim = 2,
                                      .num_attention_heads = 2,
                                      .num_key_value_heads = 2,
                                      .max_position_embeddings = 128},
                           "rope");
    ASSERT_TRUE(rope_or.ok()) << rope_or.status().ToString();
    const RoPEOutputs rope = *rope_or;
    auto cache_or = AddKVCacheUpdate(graph,
                                     0,
                                     rope.k,
                                     v,
                                     k_cache,
                                     v_cache,
                                     "kv_cache_update");
    ASSERT_TRUE(cache_or.ok()) << cache_or.status().ToString();
    const KVCachePair cache = *cache_or;

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
    const GraphValueId k_new = graph.AddConstant(cache_spec, ConstantBinding{}, "k_new");
    const GraphValueId v_new = graph.AddConstant(cache_spec, ConstantBinding{}, "v_new");
    const GraphValueId activation_cache = graph.AddInput(cache_spec, "activation_cache");
    const GraphValueId k_cache = graph.AddState(cache_spec,
                                                KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey},
                                                "k_cache");
    const GraphValueId v_cache = graph.AddState(cache_spec,
                                                KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kValue},
                                                "v_cache");

    EXPECT_FALSE(AddKVCacheUpdate(graph,
                                  0,
                                  k_new,
                                  v_new,
                                  activation_cache,
                                  v_cache,
                                  "bad_k_cache")
                         .ok());
    EXPECT_FALSE(AddKVCacheUpdate(graph,
                                  0,
                                  k_new,
                                  v_new,
                                  k_cache,
                                  activation_cache,
                                  "bad_v_cache")
                         .ok());
}

TEST(GraphOpBuilder, AddInputAndAddStateRegisterExternalValues) {
    ModelGraph graph;
    const TensorSpec input_spec = Spec(DataType::Int(64), {4});
    const TensorSpec cache_spec = Spec(DataType::Float32(), {2, 4, 2});

    const GraphValueId tokens = AddInput(graph, input_spec, "tokens");
    const StateBinding k_binding = KVCacheStateBinding{.decoder_layer_index = 0, .slot = KVCacheSlot::kKey};
    const GraphValueId k_cache = AddState(graph, cache_spec, k_binding, "k_cache");

    ASSERT_EQ(graph.GetInputs().size(), 1U);
    EXPECT_EQ(graph.GetValue(graph.GetInputs().front().value).name, "tokens");
    EXPECT_EQ(graph.GetInputs().front().value, tokens);
    EXPECT_TRUE(std::holds_alternative<StateValue>(graph.GetValue(k_cache).payload));
    const auto& state = std::get<StateValue>(graph.GetValue(k_cache).payload);
    const auto& kv_binding = std::get<KVCacheStateBinding>(state.binding);
    EXPECT_EQ(kv_binding.decoder_layer_index, 0U);
    EXPECT_EQ(kv_binding.slot, KVCacheSlot::kKey);
}

TEST(GraphOpBuilder, AddElementwiseAddSameShape) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {2, 4}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, graph.GetValue(lhs).spec);
    ASSERT_TRUE(graph.GetValue(output).producer.has_value());
    const GraphNode& node = graph.GetNode(*graph.GetValue(output).producer);
    EXPECT_EQ(node.op_type, OpType::kAdd);
    EXPECT_EQ(node.inputs.size(), 2U);
    EXPECT_EQ(node.outputs.size(), 1U);
    EXPECT_TRUE(std::holds_alternative<AddParams>(node.op_params));
}

TEST(GraphOpBuilder, AddElementwiseAddBroadcast2x1Plus1x3Produces2x3) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {2, 1}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {1, 3}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    const TensorSpec output_spec = graph.GetValue(output).spec;
    EXPECT_EQ(output_spec.shape.rank().value(), 2U);
    const auto& dims = *output_spec.shape.shape();
    ASSERT_EQ(dims.size(), 2U);
    EXPECT_EQ(dims[0].GetStaticValue(), 2);
    EXPECT_EQ(dims[1].GetStaticValue(), 3);
}

TEST(GraphOpBuilder, AddElementwiseAddRequiresMatchingDtype) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Int(32), {2, 4}), ConstantBinding{}, "rhs");

    EXPECT_FALSE(AddElementwiseAdd(graph, 0, lhs, rhs, "bad_add").ok());
}

TEST(GraphOpBuilder, AddElementwiseAddScalarPlusScalarOutputsScalar) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, graph.GetValue(lhs).spec);
    EXPECT_TRUE(graph.GetValue(output).spec.IsRankZero());
}

TEST(GraphOpBuilder, AddElementwiseAddScalarPlusTensorProducesTensorShape) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {3, 4}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, graph.GetValue(rhs).spec);
}

TEST(GraphOpBuilder, AddElementwiseAddTensorPlusScalarProducesTensorShape) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {3, 4}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    EXPECT_EQ(graph.GetValue(output).spec, graph.GetValue(lhs).spec);
}

TEST(GraphOpBuilder, AddElementwiseAddRankOnePlus3x4Produces3x4) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {1}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {3, 4}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    const TensorSpec output_spec = graph.GetValue(output).spec;
    EXPECT_FALSE(output_spec.IsRankZero());
    EXPECT_EQ(output_spec.shape.rank().value(), 2U);
    const auto& dims = *output_spec.shape.shape();
    ASSERT_EQ(dims.size(), 2U);
    EXPECT_EQ(dims[0].GetStaticValue(), 3);
    EXPECT_EQ(dims[1].GetStaticValue(), 4);
}

TEST(GraphOpBuilder, AddElementwiseAddZeroDimCompatibleWithOne) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {0}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {1}), ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    const TensorSpec output_spec = graph.GetValue(output).spec;
    EXPECT_FALSE(output_spec.IsRankZero());
    EXPECT_EQ(output_spec.shape.rank().value(), 1U);
    const auto& dims = *output_spec.shape.shape();
    ASSERT_EQ(dims.size(), 1U);
    EXPECT_EQ(dims[0].GetStaticValue(), 0);
}

TEST(GraphOpBuilder, AddElementwiseAddIncompatibleStaticShapesDeath) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(Spec(DataType::Float32(), {2, 3}), ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {4, 5}), ConstantBinding{}, "rhs");

    EXPECT_FALSE(AddElementwiseAdd(graph, 0, lhs, rhs, "bad_add").ok());
}

TEST(GraphOpBuilder, AddElementwiseAddUnrankedShapeDeath) {
    ModelGraph graph;
    const GraphValueId lhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape()}, ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(Spec(DataType::Float32(), {2, 3}), ConstantBinding{}, "rhs");

    EXPECT_FALSE(AddElementwiseAdd(graph, 0, lhs, rhs, "bad_add").ok());
}

TEST(GraphOpBuilder, AddElementwiseAddSameSymbolicDimensionPreserved) {
    ModelGraph graph;
    const auto sym = ShapeSymbol::Create();
    const GraphValueId lhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape({sym})}, ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape({sym})}, ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    const TensorSpec output_spec = graph.GetValue(output).spec;
    EXPECT_EQ(output_spec.shape.rank().value(), 1U);
    EXPECT_EQ(output_spec.shape[0], sym);
}

TEST(GraphOpBuilder, AddElementwiseAddDistinctSymbolsProduceUnknown) {
    ModelGraph graph;
    const auto sym1 = ShapeSymbol::Create();
    const auto sym2 = ShapeSymbol::Create();
    const GraphValueId lhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape({sym1})}, ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape({sym2})}, ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    const TensorSpec output_spec = graph.GetValue(output).spec;
    EXPECT_EQ(output_spec.shape.rank().value(), 1U);
    EXPECT_TRUE(output_spec.shape[0].IsUnknown());
}

TEST(GraphOpBuilder, AddElementwiseAddStaticNWithSymbolProducesN) {
    ModelGraph graph;
    const auto sym = ShapeSymbol::Create();
    const auto s3 = ShapeSymbol::CreateFromValue(3);
    const GraphValueId lhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape({s3, sym})}, ConstantBinding{}, "lhs");
    const GraphValueId rhs = graph.AddConstant(TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape({s3, ShapeSymbol::CreateFromValue(4)})}, ConstantBinding{}, "rhs");

    auto output_or = AddElementwiseAdd(graph, 0, lhs, rhs, "add");
    ASSERT_TRUE(output_or.ok()) << output_or.status().ToString();
    const GraphValueId output = *output_or;

    const TensorSpec output_spec = graph.GetValue(output).spec;
    EXPECT_EQ(output_spec.shape.rank().value(), 2U);
    EXPECT_EQ(output_spec.shape[0], s3);
    EXPECT_EQ(output_spec.shape[1].GetStaticValue(), 4);
}

}// namespace
}// namespace aethermind
