#include "aethermind/model/graph/graph.h"
#include "test_graph_helpers.h"

#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

TensorSpec ActivationSpec() {
    return Spec(DataType::Float32(), {1, 8});
}

TensorSpec TokenSpec() {
    return Spec(DataType::Int(64), {1});
}

TensorSpec WeightSpec() {
    return Spec(DataType::Float32(), {8});
}

StateBinding KvStateBinding(uint32_t decoder_layer_index = 0U) {
    return KVCacheStateBinding{.decoder_layer_index = decoder_layer_index,
                               .slot = KVCacheSlot::kKey};
}

StateBinding KStateBinding(uint32_t decoder_layer_index = 0U) {
    return KVCacheStateBinding{.decoder_layer_index = decoder_layer_index,
                               .slot = KVCacheSlot::kKey};
}

StateBinding VStateBinding(uint32_t decoder_layer_index = 0U) {
    return KVCacheStateBinding{.decoder_layer_index = decoder_layer_index,
                               .slot = KVCacheSlot::kValue};
}

AM_NODISCARD StatusOr<AddedNode> TryAddNode(ModelGraph& graph,
                                            OpType op_type,
                                            std::optional<uint32_t> decoder_layer_index,
                                            std::vector<GraphValueId> inputs,
                                            std::vector<NodeOutputDesc> outputs_desc,
                                            const OpParams& op_params = std::monostate{},
                                            std::string debug_name = {}) {
    return graph.AddNode(op_type, decoder_layer_index, std::move(inputs),
                         std::move(outputs_desc), op_params, {}, std::move(debug_name));
}

AM_NODISCARD AddedNode MustAddNode(ModelGraph& graph,
                                   OpType op_type,
                                   std::optional<uint32_t> decoder_layer_index,
                                   std::vector<GraphValueId> inputs,
                                   std::vector<NodeOutputDesc> outputs_desc,
                                   const OpParams& op_params = std::monostate{},
                                   std::string debug_name = {}) {
    auto result = TryAddNode(graph, op_type, decoder_layer_index,
                             std::move(inputs), std::move(outputs_desc), op_params,
                             std::move(debug_name));
    if (!result.ok()) {
        throw std::runtime_error("MustAddNode failed: " + result.status().ToString());
    }
    return *std::move(result);
}

GraphValueId AddEmbeddingOutput(ModelGraph& graph, std::string input_name) {
    const GraphValueId tokens = graph.AddInput(TokenSpec(), std::move(input_name));
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    return MustAddNode(graph,
                       OpType::kEmbedding,
                       std::nullopt,
                       {tokens, weight},
                       {{.payload = ActivationValue{}}},
                       EmbeddingParams{})
            .outputs.front();
}

ModelGraph BuildValidEmbeddingGraph() {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    const AddedNode embedding = MustAddNode(
            graph,
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {{.payload = ActivationValue{}}},
            EmbeddingParams{});
    graph.MarkOutput(embedding.outputs[0]);
    return graph;
}

// ---------------------------------------------------------------------------
// Existing model-graph tests (adapted to StatusOr<AddedNode> + spec-less
// NodeOutputDesc).
// ---------------------------------------------------------------------------
TEST(ModelGraph, PublicApiCreatesInputsWeightsNodesAndOutputs) {
    ModelGraph graph;

    // Embedding produces an ActivationValue that RmsNorm's kActivation port accepts.
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId emb_weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    const AddedNode emb_node = MustAddNode(
            graph,
            OpType::kEmbedding,
            std::nullopt,
            {tokens, emb_weight},
            {{.payload = ActivationValue{}, .name = "hidden"}},
            EmbeddingParams{},
            "embedding");
    const GraphValueId hidden = emb_node.outputs[0];
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            WeightBinding{.slot = ParameterSlot::kScale,
                          .decoder_layer_index = 0U,
                          .semantic_role = TransformerWeightRole::kInputNorm});
    const AddedNode node = MustAddNode(
            graph,
            OpType::kRmsNorm,
            0U,
            {hidden, weight},
            {{.payload = ActivationValue{}, .name = "normed_output"}},
            RmsNormParams{},
            "rms_norm_0");
    graph.MarkOutput(node.outputs[0]);

    ASSERT_EQ(graph.GetInputs().size(), 1U);
    EXPECT_EQ(graph.GetInputs()[0].value, tokens);
    EXPECT_EQ(graph.GetValue(graph.GetInputs()[0].value).name, "token_ids");
    ASSERT_EQ(graph.GetOutputs().size(), 1U);
    EXPECT_EQ(graph.GetOutputs()[0].value, node.outputs[0]);
    EXPECT_EQ(graph.GetValue(graph.GetOutputs()[0].value).name, "normed_output");
    ASSERT_EQ(graph.GetNodes().size(), 2U);
    EXPECT_EQ(graph.GetNode(node.node).op_type, OpType::kRmsNorm);
    EXPECT_EQ(graph.GetNode(node.node).name, "rms_norm_0");
    ASSERT_GE(graph.GetValues().size(), 4U);
    EXPECT_TRUE(std::holds_alternative<ModelInputValue>(graph.GetValue(tokens).payload));
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
    const GraphValueId input = AddEmbeddingOutput(graph, "tokens");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            WeightBinding{.slot = ParameterSlot::kScale,
                          .decoder_layer_index = 0U,
                          .semantic_role = TransformerWeightRole::kInputNorm});

    const AddedNode node = MustAddNode(
            graph,
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{.eps = 2.5e-3F});

    const auto* params = std::get_if<RmsNormParams>(&graph.GetNode(node.node).op_params);
    ASSERT_NE(params, nullptr);
    EXPECT_FLOAT_EQ(params->eps, 2.5e-3F);
}

TEST(ModelGraph, ValidateRejectsMissingOpParams) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    const auto result = TryAddNode(
            graph,
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {{.payload = ActivationValue{}}});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsWrongOpParamsForOpType) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    const auto result = TryAddNode(
            graph,
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidRmsNormEps) {
    ModelGraph graph;
    const GraphValueId input = AddEmbeddingOutput(graph, "tokens");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            WeightBinding{.slot = ParameterSlot::kScale,
                          .semantic_role = TransformerWeightRole::kInputNorm});
    const auto result = TryAddNode(
            graph,
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{.eps = 0.0F});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateAcceptsSiluNodeWithSiluParams) {
    ModelGraph graph;
    const GraphValueId input = AddEmbeddingOutput(graph, "tokens");
    const AddedNode node = MustAddNode(
            graph,
            OpType::kSilu,
            0U,
            {input},
            {{.payload = ActivationValue{}}},
            SiluParams{});
    graph.MarkOutput(node.outputs[0]);

    const Status status = graph.Validate();
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelGraph, ValidateRejectsSiluNodeWithWrongParams) {
    ModelGraph graph;
    const GraphValueId input = AddEmbeddingOutput(graph, "tokens");
    const auto result = TryAddNode(
            graph,
            OpType::kSilu,
            0U,
            {input},
            {{.payload = ActivationValue{}}},
            EmbeddingParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateAcceptsElementwiseMulNodeWithElementwiseMulParams) {
    ModelGraph graph;
    const GraphValueId lhs = AddEmbeddingOutput(graph, "tokens_lhs");
    const GraphValueId rhs = AddEmbeddingOutput(graph, "tokens_rhs");
    const AddedNode node = MustAddNode(
            graph,
            OpType::kElementwiseMul,
            0U,
            {lhs, rhs},
            {{.payload = ActivationValue{}}},
            ElementwiseMulParams{});
    graph.MarkOutput(node.outputs[0]);

    const Status status = graph.Validate();
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelGraph, ValidateRejectsElementwiseMulNodeWithWrongParams) {
    ModelGraph graph;
    const GraphValueId lhs = AddEmbeddingOutput(graph, "tokens_lhs");
    const GraphValueId rhs = AddEmbeddingOutput(graph, "tokens_rhs");
    const auto result = TryAddNode(
            graph,
            OpType::kElementwiseMul,
            0U,
            {lhs, rhs},
            {{.payload = ActivationValue{}}},
            AddParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidRoPEParams) {
    ModelGraph graph;
    const GraphValueId q = graph.AddInput(ActivationSpec(), "q");
    const GraphValueId k = graph.AddInput(ActivationSpec(), "k");
    const GraphValueId position_ids = graph.AddInput(ActivationSpec(), "position_ids");
    const auto result = TryAddNode(
            graph,
            OpType::kRoPE,
            0U,
            {q, k, position_ids},
            {{.payload = ActivationValue{}},
             {.payload = ActivationValue{}}},
            RoPEParams{.head_dim = 0,
                       .num_attention_heads = 4,
                       .num_key_value_heads = 2,
                       .max_position_embeddings = 128});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsMonostatePayload) {
    ModelGraph graph({}, {}, {GraphValue{}});
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsActivationWithoutProducer) {
    ModelGraph graph({}, {},
                     {GraphValue{.payload = ActivationValue{},
                                 .spec = ActivationSpec(),
                                 .name = ""}});
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsExternalValueWithProducer) {
    std::vector<GraphValue> values = {
            {.payload = ModelInputValue{},
             .spec = TokenSpec(),
             .producer = GraphNodeId{0},
             .name = ""},
    };

    ModelGraph graph({}, {}, std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, PublicApiCreatesStateValues) {
    ModelGraph graph;
    const GraphValueId state = graph.AddState(
            ActivationSpec(),
            KvStateBinding(),
            "kv_cache");

    ASSERT_EQ(graph.GetValues().size(), 1U);
    const GraphValue& value = graph.GetValue(state);
    ASSERT_TRUE(std::holds_alternative<StateValue>(value.payload));
    const auto* binding = std::get_if<KVCacheStateBinding>(
            &std::get<StateValue>(value.payload).binding);
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->decoder_layer_index, 0U);
    EXPECT_EQ(binding->slot, KVCacheSlot::kKey);
    EXPECT_FALSE(value.producer.has_value());
}

TEST(ModelGraph, PublicApiCreatesConstantValues) {
    ModelGraph graph;
    auto inline_data = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}});
    const GraphValueId constant = graph.AddConstant(
            Spec(DataType::Float32(), {1}),
            ConstantBinding{.inline_data = std::move(inline_data),
                            .name = "rope.sin_cos_table"},
            "rope_table");

    ASSERT_EQ(graph.GetValues().size(), 1U);
    const GraphValue& value = graph.GetValue(constant);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(value.payload));
    const ConstantBinding& binding = std::get<ConstantValue>(value.payload).binding;
    EXPECT_EQ(binding.name, "rope.sin_cos_table");
    ASSERT_TRUE(binding.inline_data != nullptr);
    EXPECT_EQ(binding.inline_data->size(), 3U);
    EXPECT_FALSE(value.producer.has_value());
    EXPECT_EQ(value.name, "rope_table");
}

TEST(ModelGraph, SetQuantizationAttachesSchemeToValue) {
    ModelGraph graph;
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding},
            "embed.weight");

    graph.SetQuantization(weight,
                          {
                                  .kind = QuantizationKind::kInt4,
                                  .group_size = 64,
                                  .scale_dtype = DataType::Float32(),
                                  .has_zero_point = true,
                          });

    const GraphValue& value = graph.GetValue(weight);
    EXPECT_EQ(value.quantization.kind, QuantizationKind::kInt4);
    EXPECT_EQ(value.quantization.group_size, 64U);
    EXPECT_TRUE(value.quantization.has_zero_point);
}

TEST(ModelGraph, ValidateAcceptsExternalConstantValue) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "tokens");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding});
    UNUSED(graph.AddConstant(
            Spec(DataType::Float32(), {8}),
            {.name = "embed.bias"}));
    const AddedNode embedding = MustAddNode(
            graph,
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {{.payload = ActivationValue{}}},
            EmbeddingParams{});
    graph.MarkOutput(embedding.outputs[0]);

    EXPECT_TRUE(graph.Validate().ok());
}

TEST(ModelGraph, ValidateRejectsConstantValueWithProducer) {
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{.binding = ConstantBinding{.name = "c"}},
             .spec = Spec(DataType::Float32(), {1}),
             .producer = GraphNodeId{0},
             .name = {}},
    };
    ModelGraph graph({}, {}, std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateAcceptsStateUpdateNode) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "tokens");
    const GraphValueId embedding_weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            {.slot = ParameterSlot::kEmbeddingTable,
             .semantic_role = TransformerWeightRole::kTokenEmbedding});
    const GraphValueId hidden = MustAddNode(
                                        graph,
                                        OpType::kEmbedding,
                                        std::nullopt,
                                        {tokens, embedding_weight},
                                        {{.payload = ActivationValue{}}},
                                        EmbeddingParams{})
                                        .outputs.front();
    const GraphValueId k_weight = graph.AddWeight(
            Spec(DataType::Float32(), {8, 8}),
            {.slot = ParameterSlot::kKernel,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kAttentionK});
    const GraphValueId v_weight = graph.AddWeight(
            Spec(DataType::Float32(), {8, 8}),
            {.slot = ParameterSlot::kKernel,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kAttentionV});
    const GraphValueId k = MustAddNode(
                                   graph,
                                   OpType::kLinear,
                                   std::nullopt,
                                   {hidden, k_weight},
                                   {{.payload = ActivationValue{}}},
                                   LinearParams{})
                                   .outputs.front();
    const GraphValueId v = MustAddNode(
                                   graph,
                                   OpType::kLinear,
                                   std::nullopt,
                                   {hidden, v_weight},
                                   {{.payload = ActivationValue{}}},
                                   LinearParams{})
                                   .outputs.front();
    const GraphValueId k_cache_in = graph.AddState(
            ActivationSpec(),
            KStateBinding(0U),
            "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(
            ActivationSpec(),
            VStateBinding(0U),
            "v_cache_in");

    const AddedNode update = MustAddNode(
            graph,
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {{.payload = StateValue{.binding = KStateBinding(0U)}},
             {.payload = StateValue{.binding = VStateBinding(0U)}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_NE(k_cache_in, update.outputs[0]);
    EXPECT_NE(v_cache_in, update.outputs[1]);
    const GraphValue& k_cache_out = graph.GetValue(update.outputs[0]);
    ASSERT_TRUE(std::holds_alternative<StateValue>(k_cache_out.payload));
    ASSERT_TRUE(k_cache_out.producer.has_value());
    EXPECT_EQ(*k_cache_out.producer, update.node);
    const GraphValue& v_cache_out = graph.GetValue(update.outputs[1]);
    ASSERT_TRUE(std::holds_alternative<StateValue>(v_cache_out.payload));
    ASSERT_TRUE(v_cache_out.producer.has_value());
    EXPECT_EQ(*v_cache_out.producer, update.node);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateAcrossDecoderLayerStateFamilies) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId layer0_k_cache = graph.AddState(
            ActivationSpec(), KStateBinding(0U), "layer0_k_cache");
    const GraphValueId layer0_v_cache = graph.AddState(
            ActivationSpec(), VStateBinding(0U), "layer0_v_cache");

    const auto result = TryAddNode(
            graph,
            OpType::kKVCacheUpdate,
            1U,
            {k, v, layer0_k_cache, layer0_v_cache},
            {{.payload = StateValue{.binding = KStateBinding(1U)}},
             {.payload = StateValue{.binding = VStateBinding(1U)}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateKStateFamilyMismatch) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(
            ActivationSpec(), KStateBinding(0U), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(
            ActivationSpec(), VStateBinding(0U), "v_cache_in");

    const auto result = TryAddNode(
            graph,
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {{.payload = StateValue{.binding = KStateBinding(1U)}},
             {.payload = StateValue{.binding = VStateBinding(0U)}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateVStateFamilyMismatch) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(
            ActivationSpec(), KStateBinding(0U), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(
            ActivationSpec(), VStateBinding(0U), "v_cache_in");

    const auto result = TryAddNode(
            graph,
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {{.payload = StateValue{.binding = KStateBinding(0U)}},
             {.payload = StateValue{.binding = VStateBinding(1U)}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateWithSwappedKeyValueStateSlots) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId wrong_k_cache_in = graph.AddState(
            ActivationSpec(), VStateBinding(0U), "wrong_k_cache_in");
    const GraphValueId wrong_v_cache_in = graph.AddState(
            ActivationSpec(), KStateBinding(0U), "wrong_v_cache_in");

    const auto result = TryAddNode(
            graph,
            OpType::kKVCacheUpdate,
            0U,
            {k, v, wrong_k_cache_in, wrong_v_cache_in},
            {{.payload = StateValue{.binding = VStateBinding(0U)}},
             {.payload = StateValue{.binding = KStateBinding(0U)}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateWithMixedKeyValueStateLayers) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(
            ActivationSpec(), KStateBinding(0U), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(
            ActivationSpec(), VStateBinding(1U), "v_cache_in");

    const auto result = TryAddNode(
            graph,
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {{.payload = StateValue{.binding = KStateBinding(0U)}},
             {.payload = StateValue{.binding = VStateBinding(1U)}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsAttentionWithSwappedKeyValueStateSlots) {
    ModelGraph graph;
    const GraphValueId q = AddEmbeddingOutput(graph, "tokens_q");
    const GraphValueId wrong_k_cache = graph.AddState(
            ActivationSpec(), VStateBinding(0U), "wrong_k_cache");
    const GraphValueId wrong_v_cache = graph.AddState(
            ActivationSpec(), KStateBinding(0U), "wrong_v_cache");

    const auto result = TryAddNode(
            graph,
            OpType::kAttention,
            0U,
            {q, wrong_k_cache, wrong_v_cache},
            {{.payload = ActivationValue{}}},
            AttentionParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsAttentionWithWrongCacheLayer) {
    ModelGraph graph;
    const GraphValueId q = AddEmbeddingOutput(graph, "tokens_q");
    const GraphValueId k_cache = graph.AddState(
            ActivationSpec(), KStateBinding(1U), "k_cache");
    const GraphValueId v_cache = graph.AddState(
            ActivationSpec(), VStateBinding(1U), "v_cache");

    const auto result = TryAddNode(
            graph,
            OpType::kAttention,
            0U,
            {q, k_cache, v_cache},
            {{.payload = ActivationValue{}}},
            AttentionParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsStateValueWithInvalidProducer) {
    std::vector<GraphValue> values = {
            {.payload = StateValue{.binding = KvStateBinding()},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = ""},
    };
    ModelGraph graph({}, {}, std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidKvCacheStateBindingSlot) {
    ModelGraph graph;
    const GraphValueId state = graph.AddState(
            ActivationSpec(),
            KVCacheStateBinding{.decoder_layer_index = 0U,
                                .slot = static_cast<KVCacheSlot>(255)},
            "kv_cache");
    EXPECT_EQ(state.index, 0U);

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsStateValueAsGraphOutput) {
    ModelGraph graph;
    const GraphValueId state = graph.AddState(
            ActivationSpec(), KvStateBinding(), "kv_cache");
    graph.MarkOutput(state);

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateWithActivationOutput) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(
            ActivationSpec(), KStateBinding(), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(
            ActivationSpec(), VStateBinding(), "v_cache_in");

    const auto result = TryAddNode(
            graph,
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {{},
             {.payload = StateValue{.binding = VStateBinding()}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsNodeOutputReusingInputValue) {
    std::vector<GraphValue> values = {
            {.payload = ModelInputValue{}, .spec = TokenSpec(), .name = ""},
            {.payload = WeightValue{
                     .binding = {.slot = ParameterSlot::kEmbeddingTable,
                                 .semantic_role = TransformerWeightRole::kTokenEmbedding}},
             .spec = Spec(DataType::Float32(), {32, 8}),
             .name = ""},
            {.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}, .name = ""},
            {.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{1}, .name = ""},
            {.payload = StateValue{.binding = KStateBinding()}, .spec = ActivationSpec(), .producer = GraphNodeId{2}, .name = ""},
            {.payload = StateValue{.binding = VStateBinding()}, .spec = ActivationSpec(), .producer = GraphNodeId{2}, .name = ""},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kEmbedding,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = EmbeddingParams{},
             .name = ""},
            {.op_type = OpType::kEmbedding,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{3}},
             .op_params = EmbeddingParams{},
             .name = ""},
            {.op_type = OpType::kKVCacheUpdate,
             .inputs = {GraphValueId{2},
                        GraphValueId{3},
                        GraphValueId{4},
                        GraphValueId{5}},
             .outputs = {GraphValueId{4}, GraphValueId{5}},
             .op_params = KVCacheUpdateParams{},
             .name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsSchemaPortKindMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ModelInputValue{}, .spec = TokenSpec(), .name = ""},
            {.payload = ActivationValue{},
             .spec = WeightSpec(),
             .producer = GraphNodeId{0},
             .name = ""},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = ""},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kEmbedding,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsSchemaArityMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ModelInputValue{}, .spec = TokenSpec(), .name = ""},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = ""},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kEmbedding,
             .inputs = {GraphValueId{0}},
             .outputs = {GraphValueId{1}},
             .name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsOutputOnExternalValue) {
    ModelGraph graph;
    const GraphValueId activation = graph.AddConstant(
            ActivationSpec(), ConstantBinding{}, "act");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});
    UNUSED(MustAddNode(
            graph,
            OpType::kRmsNorm,
            0U,
            {activation, weight},
            {{.payload = ActivationValue{}, .name = ""}},
            RmsNormParams{}));
    const GraphValueId external = graph.AddInput(ActivationSpec(), "external");
    graph.MarkOutput(external);

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsWeightSlotMismatchWithOperator) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), ConstantBinding{}, "act");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {8, 8}),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});
    const auto result = TryAddNode(
            graph, OpType::kLinear, 0U, {input, weight},
            {{.payload = ActivationValue{}}},
            LinearParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsWeightSlotMismatchWithSemanticRole) {
    ModelGraph graph;
    UNUSED(graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kAttentionQ}));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsPerLayerRoleWithoutDecoderLayerIndex) {
    ModelGraph graph;
    UNUSED(graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kKernel,
             .semantic_role = TransformerWeightRole::kAttentionQ}));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsModelLevelRoleWithDecoderLayerIndex) {
    ModelGraph graph;
    UNUSED(graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            {.slot = ParameterSlot::kEmbeddingTable,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kTokenEmbedding}));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateAcceptsMonostateSemanticRoleForGenericGraph) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "tokens");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {32, 8}),
            {.slot = ParameterSlot::kEmbeddingTable});
    const AddedNode embedding = MustAddNode(
            graph,
            OpType::kEmbedding, std::nullopt, {tokens, weight},
            {{.payload = ActivationValue{}}},
            EmbeddingParams{});
    graph.MarkOutput(embedding.outputs[0]);

    const Status status = graph.Validate();
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelGraph, BuildConsumerIndexMapsConsumers) {
    ModelGraph graph;
    const GraphValueId input = AddEmbeddingOutput(graph, "tokens");
    const AddedNode softmax = MustAddNode(
            graph,
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {{.payload = ActivationValue{}, .name = ""}},
            SoftmaxParams{});
    const AddedNode argmax = MustAddNode(
            graph,
            OpType::kArgmax,
            std::nullopt,
            {softmax.outputs[0]},
            {{.payload = ActivationValue{}, .name = ""}},
            ArgmaxParams{.axis = -1});
    const AddedNode another = MustAddNode(
            graph,
            OpType::kSoftmax,
            std::nullopt,
            {softmax.outputs[0]},
            {{.payload = ActivationValue{}, .name = ""}},
            SoftmaxParams{});

    const auto index = BuildConsumerIndex(graph);
    ASSERT_TRUE(index.ok()) << index.status().ToString();
    ASSERT_EQ(index->size(), graph.GetValues().size());
    const auto input_consumers = GetConsumers(*index, input);
    ASSERT_EQ(input_consumers.size(), 1U);
    EXPECT_EQ(input_consumers[0], softmax.node);
    const auto softmax_consumers = GetConsumers(*index, softmax.outputs[0]);
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
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), ConstantBinding{}, "input");
    const AddedNode softmax = MustAddNode(
            graph,
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {{.payload = ActivationValue{}, .name = ""}},
            SoftmaxParams{});
    const AddedNode argmax = MustAddNode(
            graph,
            OpType::kArgmax,
            std::nullopt,
            {softmax.outputs[0]},
            {{.payload = ActivationValue{}, .name = ""}},
            ArgmaxParams{.axis = -1});

    const auto order = graph.TopologicalOrder();

    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), 2U);
    EXPECT_EQ((*order)[0], softmax.node);
    EXPECT_EQ((*order)[1], argmax.node);
}

TEST(ModelGraph, TopologicalOrderRejectsActivationCycle) {
    std::vector<GraphValue> values = {
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = ""},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{1},
             .name = ""},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kSoftmax,
             .inputs = {GraphValueId{1}},
             .outputs = {GraphValueId{0}},
             .name = ""},
            {.op_type = OpType::kSoftmax,
             .inputs = {GraphValueId{0}},
             .outputs = {GraphValueId{1}},
             .name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const auto order = graph.TopologicalOrder();

    ASSERT_FALSE(order.ok());
    EXPECT_EQ(order.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, FindNodesByOpTypeReturnsMatchingIdsInIndexOrder) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), ConstantBinding{}, "input");
    const AddedNode softmax_a = MustAddNode(
            graph,
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {{.payload = ActivationValue{}}},
            SoftmaxParams{});
    const AddedNode argmax = MustAddNode(
            graph,
            OpType::kArgmax,
            std::nullopt,
            {softmax_a.outputs[0]},
            {{.payload = ActivationValue{}}},
            ArgmaxParams{.axis = -1});
    const AddedNode softmax_b = MustAddNode(
            graph,
            OpType::kSoftmax,
            std::nullopt,
            {softmax_a.outputs[0]},
            {{.payload = ActivationValue{}}},
            SoftmaxParams{});

    const auto softmax_ids = graph.FindNodesByOpType(OpType::kSoftmax);

    ASSERT_EQ(softmax_ids.size(), 2U);
    EXPECT_EQ(softmax_ids[0], softmax_a.node);
    EXPECT_EQ(softmax_ids[1], softmax_b.node);
    EXPECT_EQ(graph.GetNode(softmax_ids[0]).op_type, OpType::kSoftmax);
}

TEST(ModelGraph, FindNodesByOpTypeReturnsEmptyWhenNoMatch) {
    ModelGraph graph;
    const GraphValueId input = AddEmbeddingOutput(graph, "tokens");
    UNUSED(MustAddNode(
            graph,
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            SoftmaxParams{}));

    const std::vector<GraphNodeId> ids = graph.FindNodesByOpType(OpType::kAttention);
    EXPECT_TRUE(ids.empty());
}

TEST(ModelGraph, FindNodesByOpTypeReturnsEmptyForEmptyGraph) {
    ModelGraph graph;
    const std::vector<GraphNodeId> ids = graph.FindNodesByOpType(OpType::kSoftmax);
    EXPECT_TRUE(ids.empty());
}

// ===========================================================================
// ModelGraphSemanticValidation
// ===========================================================================
TEST(ModelGraphSemanticValidation, OutputPreservesNonsemanticMetadata) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), {}, "input");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{
                    .payload = ActivationValue{},
                    .quantization = {.kind = QuantizationKind::kInt8, .group_size = 32},
                    .name = "named_output",
            }},
            RmsNormParams{});

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& out_value = graph.GetValue((*result).outputs[0]);
    EXPECT_EQ(out_value.quantization.kind, QuantizationKind::kInt8);
    EXPECT_EQ(out_value.quantization.group_size, 32U);
    EXPECT_EQ(out_value.name, "named_output");
}

TEST(ModelGraphSemanticValidation, AddNodeRejectsInvalidInputId) {
    ModelGraph graph;
    UNUSED(graph.AddConstant(
            ActivationSpec(), ConstantBinding{}, "input"));
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const size_t node_count_before = graph.GetNodes().size();
    const size_t value_count_before = graph.GetValues().size();

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {GraphValueId{99}, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    const std::string msg = result.status().ToString();
    EXPECT_NE(msg.find("invalid value id"), std::string::npos);
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_EQ(graph.GetNodes().size(), node_count_before);
    EXPECT_EQ(graph.GetValues().size(), value_count_before);
}

TEST(ModelGraphSemanticValidation, AddNodeRejectsOutputCountMismatch) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), {}, "input");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const size_t node_count_before = graph.GetNodes().size();
    const size_t value_count_before = graph.GetValues().size();

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}},
             {.payload = ActivationValue{}}},
            RmsNormParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    const std::string msg = result.status().ToString();
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_EQ(graph.GetNodes().size(), node_count_before);
    EXPECT_EQ(graph.GetValues().size(), value_count_before);
}

TEST(ModelGraphSemanticValidation, AddNodeRejectsInputCountMismatch) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), {}, "input");

    const size_t node_count_before = graph.GetNodes().size();
    const size_t value_count_before = graph.GetValues().size();

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            RmsNormParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(graph.GetNodes().size(), node_count_before);
    EXPECT_EQ(graph.GetValues().size(), value_count_before);
}

TEST(ModelGraphSemanticValidation, AddNodeRejectsBadDtype) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            Spec(DataType::Float32(), {1, 8}), {}, "input");
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Int(8), {8}),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            RmsNormParams{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    const std::string msg = result.status().ToString();
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, AddNodePreservesDebugName) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), {}, "input");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{},
            {},
            "rms_norm");

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(graph.GetNode(result->node).name, "rms_norm");
}

TEST(ModelGraphSemanticValidation, ValidateRejectsForgedOutputSpec) {
    std::vector<GraphValue> values = {
            {.payload = ActivationValue{}, .spec = ActivationSpec(), .name = "input"},
            {.payload = WeightValue{.binding = {.slot = ParameterSlot::kScale,
                                                .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = WeightSpec(),
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = Spec(DataType::Float32(),
                          {999, 999}),
             .producer = GraphNodeId{0},
             .name = "forged"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .name = "bad"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStaleRuntimeCheckCondition) {
    // Build via AddNode with symbolic shapes to discover the derived metadata,
    // including one deferred DimEqualConstraint. Then tamper the stored condition
    // and verify Validate detects it.
    ModelGraph gold;
    const auto sym_input_spec = TensorSpec{
            DataType::Float32(),
            SymbolicShape{ShapeSymbol::CreateFromValue(1), ShapeSymbol::Create()}};
    const auto sym_weight_spec = TensorSpec{
            DataType::Float32(),
            SymbolicShape{ShapeSymbol::Create()}};
    const GraphValueId gold_input = gold.AddConstant(
            sym_input_spec, {}, "input");
    const GraphValueId gold_weight = gold.AddWeight(
            sym_weight_spec,
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});
    const auto gold_result = gold.AddNode(
            OpType::kRmsNorm, 0U, {gold_input, gold_weight},
            {{.payload = ActivationValue{}}}, RmsNormParams{});
    ASSERT_TRUE(gold_result.ok());
    const auto& gold_node = gold.GetNode((*gold_result).node);
    ASSERT_EQ(gold_node.runtime_checks.size(), 1U);
    const auto& gold_check = gold_node.runtime_checks[0];

    // Tamper: change the stored condition but keep the error_context.
    std::vector<ShapeConstraint> tampered_checks = {gold_check};
    tampered_checks[0].condition = RankEqualConstraint{
            .port = {TensorPortType::kInput, 0},
            .target_rank = 99};

    std::vector<GraphValue> values = {
            {.payload = ConstantValue{ConstantBinding{}},
             .spec = sym_input_spec,
             .name = "input"},
            {.payload = WeightValue{.binding = {.slot = ParameterSlot::kScale,
                                                .decoder_layer_index = 0U,
                                                .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = sym_weight_spec,
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = sym_input_spec,
             .producer = GraphNodeId{0},
             .name = "out"},
    };

    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .runtime_checks = std::move(tampered_checks),
             .name = "bad"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();
    ASSERT_FALSE(status.ok()) << status.ToString();
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("stale runtime check"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsSameConditionWithStaleErrorContext) {
    // Build via AddNode with symbolic shapes to discover the derived metadata.
    // Then tamper only the error_context (keep the condition exactly as derived)
    // and verify Validate detects the mismatch.
    ModelGraph gold;
    const auto sym_input_spec = TensorSpec{
            DataType::Float32(),
            SymbolicShape{ShapeSymbol::CreateFromValue(1), ShapeSymbol::Create()}};
    const auto sym_weight_spec = TensorSpec{
            DataType::Float32(),
            SymbolicShape{ShapeSymbol::Create()}};
    const GraphValueId gold_input = gold.AddConstant(
            sym_input_spec, {}, "input");
    const GraphValueId gold_weight = gold.AddWeight(
            sym_weight_spec,
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});
    const auto gold_result = gold.AddNode(
            OpType::kRmsNorm, 0U, {gold_input, gold_weight},
            {{.payload = ActivationValue{}}}, RmsNormParams{});
    ASSERT_TRUE(gold_result.ok());
    const auto& gold_node = gold.GetNode((*gold_result).node);
    ASSERT_EQ(gold_node.runtime_checks.size(), 1U);
    const auto& gold_check = gold_node.runtime_checks[0];

    // Tamper only the error_context; condition stays identical.
    std::vector<ShapeConstraint> tampered_checks = {gold_check};
    tampered_checks[0].error_context = "stale_context";

    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}}, .spec = sym_input_spec, .name = "input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kScale,
                                             .decoder_layer_index = 0U,
                                             .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = sym_weight_spec,
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = sym_input_spec,
             .producer = GraphNodeId{0},
             .name = "out"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .runtime_checks = std::move(tampered_checks),
             .name = "mismatch_ctx"}};
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("stale runtime check"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredOutputCountMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ActivationValue{}, .spec = ActivationSpec(), .name = "input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kScale,
                                             .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = WeightSpec(),
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "out0"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "out1_extra"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}, GraphValueId{3}},
             .op_params = RmsNormParams{},
             .name = "bad"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraphSemanticValidation, AddNodeErrorsContainOpContext) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}}},
            EmbeddingParams{});

    EXPECT_FALSE(result.ok());
    const std::string msg = result.status().ToString();
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_NE(msg.find("node 0"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, AddNodeErrorIncludesInputPortName) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), {}, "input");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kKernel,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kAttentionK});

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{});
    EXPECT_FALSE(result.ok());
    const std::string msg = result.status().ToString();
    EXPECT_NE(msg.find("input[1]"), std::string::npos);
    EXPECT_NE(msg.find("weight slot mismatch"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, AddNodeErrorIncludesOutputPortName) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(
            ActivationSpec(), KStateBinding(), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(
            ActivationSpec(), VStateBinding(), "v_cache_in");

    // KVCacheUpdate expects both outputs to be State; providing Activation should fail.
    const auto result = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {{.payload = ActivationValue{}}, {.payload = ActivationValue{}}},
            KVCacheUpdateParams{});

    EXPECT_FALSE(result.ok());
    const std::string msg = result.status().ToString();
    EXPECT_NE(msg.find("output"), std::string::npos);
    EXPECT_NE(msg.find("KVCacheUpdate"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredNodeWithWrongParams) {
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}},
             .spec = ActivationSpec(),
             .name = "input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kScale,
                                             .decoder_layer_index = 0U,
                                             .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = WeightSpec(),
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "out"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = EmbeddingParams{},
             .name = "my_rms"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("node 0"), std::string::npos);
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_NE(msg.find("my_rms"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredNodeWithInputPayloadMismatch) {
    std::vector<GraphValue> values = {
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kEmbeddingTable,
                                             .semantic_role = TransformerWeightRole::kTokenEmbedding}},
             .spec = ActivationSpec(),
             .name = "bad_input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kScale,
                                             .decoder_layer_index = 0U,
                                             .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = WeightSpec(),
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "out"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .name = "bad_rms"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("node 0"), std::string::npos);
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_NE(msg.find("bad_rms"), std::string::npos);
    EXPECT_NE(msg.find("input[0]"), std::string::npos);
    EXPECT_NE(msg.find("input"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredNodeWithWeightSlotMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}}, .spec = ActivationSpec(), .name = "input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kKernel,
                                             .decoder_layer_index = 0U,
                                             .semantic_role = TransformerWeightRole::kAttentionK}},
             .spec = WeightSpec(),
             .name = "bad_weight"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "out"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .name = "slot_mismatch"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("node 0"), std::string::npos);
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_NE(msg.find("slot_mismatch"), std::string::npos);
    EXPECT_NE(msg.find("input[1]"), std::string::npos);
    EXPECT_NE(msg.find("weight"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredNodeWithOutputPayloadMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}}, .spec = ActivationSpec(), .name = "k"},
            {.payload = ConstantValue{{}}, .spec = ActivationSpec(), .name = "v"},
            {.payload = StateValue{.binding = KStateBinding()},
             .spec = ActivationSpec(),
             .name = "k_cache_in"},
            {.payload = StateValue{.binding = VStateBinding()},
             .spec = ActivationSpec(),
             .name = "v_cache_in"},
            {.payload = StateValue{.binding = KStateBinding()},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "k_cache_out"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "bad_v_out"},
    };

    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kKVCacheUpdate,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1},
                        GraphValueId{2}, GraphValueId{3}},
             .outputs = {GraphValueId{4}, GraphValueId{5}},
             .op_params = KVCacheUpdateParams{},
             .name = "kv_update"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("node 0"), std::string::npos);
    EXPECT_NE(msg.find("KVCacheUpdate"), std::string::npos);
    EXPECT_NE(msg.find("kv_update"), std::string::npos);
    EXPECT_NE(msg.find("output"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredNodeWithArityMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}},
             .spec = ActivationSpec(),
             .name = "input"},
            {.payload = ActivationValue{},
             .spec = ActivationSpec(),
             .producer = GraphNodeId{0},
             .name = "out"},
    };

    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}},
             .outputs = {GraphValueId{1}},
             .op_params = RmsNormParams{},
             .name = "arity_mismatch"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("node 0"), std::string::npos);
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_NE(msg.find("arity_mismatch"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, ValidateRejectsStoredNodeWithOutputSpecMismatch) {
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}},
             .spec = ActivationSpec(),
             .name = "input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kScale,
                                             .decoder_layer_index = 0U,
                                             .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = WeightSpec(),
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = Spec(DataType::Float32(), {999, 999}),
             .producer = GraphNodeId{0},
             .name = "forged"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .name = "bad_spec"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));
    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    const std::string msg = status.ToString();
    EXPECT_NE(msg.find("node 0"), std::string::npos);
    EXPECT_NE(msg.find("RmsNorm"), std::string::npos);
    EXPECT_NE(msg.find("bad_spec"), std::string::npos);
    EXPECT_NE(msg.find("output[0]"), std::string::npos);
    EXPECT_NE(msg.find("spec mismatch"), std::string::npos);
}

TEST(ModelGraphSemanticValidation, DerivedOutputSpecIsExactAndNotUserSupplied) {
    ModelGraph graph;
    const GraphValueId input = graph.AddConstant(
            ActivationSpec(), {}, "input");
    const GraphValueId weight = graph.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});

    const auto result = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {{.payload = ActivationValue{}}},
            RmsNormParams{});

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const GraphValue& out_value = graph.GetValue((*result).outputs[0]);
    EXPECT_EQ(out_value.spec.dtype, DataType::Float32());
    EXPECT_TRUE(out_value.spec.shape.IsRanked());
}

TEST(ModelGraphSemanticValidation, RoPENodeHasDistinctOutputIds) {
    ModelGraph graph;
    const GraphValueId q = graph.AddConstant(ActivationSpec(), ConstantBinding{}, "q");
    const GraphValueId k = graph.AddConstant(ActivationSpec(), ConstantBinding{}, "k");
    const GraphValueId position_ids = graph.AddInput(TokenSpec(), "position_ids");

    const auto result = graph.AddNode(
            OpType::kRoPE,
            0U,
            {q, k, position_ids},
            {{.payload = ActivationValue{}},
             {.payload = ActivationValue{}}},
            RoPEParams{.head_dim = 64,
                       .num_attention_heads = 4,
                       .num_key_value_heads = 2,
                       .max_position_embeddings = 128});

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const AddedNode& added = *result;
    ASSERT_EQ(added.outputs.size(), 2U);
    EXPECT_NE(added.outputs[0], added.outputs[1]);
    const GraphValue& out0 = graph.GetValue(added.outputs[0]);
    const GraphValue& out1 = graph.GetValue(added.outputs[1]);
    EXPECT_EQ(out0.spec.dtype, DataType::Float32());
    EXPECT_EQ(out1.spec.dtype, DataType::Float32());
    ASSERT_TRUE(out0.producer.has_value());
    ASSERT_TRUE(out1.producer.has_value());
    EXPECT_EQ(*out0.producer, added.node);
    EXPECT_EQ(*out1.producer, added.node);
}

TEST(ModelGraphSemanticValidation, KVCacheUpdateOutputHasStatePayloadAndDerivedSpec) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_in = graph.AddState(ActivationSpec(), KStateBinding(), "k_in");
    const GraphValueId v_in = graph.AddState(ActivationSpec(), VStateBinding(), "v_in");

    const auto result = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_in, v_in},
            {{.payload = StateValue{.binding = KStateBinding()}},
             {.payload = StateValue{.binding = VStateBinding()}}},
            KVCacheUpdateParams{});

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const AddedNode& added = *result;
    ASSERT_EQ(added.outputs.size(), 2U);
    EXPECT_NE(added.outputs[0], added.outputs[1]);
    const GraphValue& out0 = graph.GetValue(added.outputs[0]);
    const GraphValue& out1 = graph.GetValue(added.outputs[1]);
    EXPECT_TRUE(std::holds_alternative<StateValue>(out0.payload));
    EXPECT_TRUE(std::holds_alternative<StateValue>(out1.payload));
    EXPECT_EQ(out0.spec.dtype, DataType::Float32());
    EXPECT_EQ(out1.spec.dtype, DataType::Float32());
}

TEST(ModelGraphSemanticValidation, RawConstructorAcceptsExactDerivedMetadata) {
    // Build via AddNode first to discover the exact derived metadata.
    ModelGraph gold;
    const GraphValueId gold_input = gold.AddConstant(
            ActivationSpec(), {}, "input");
    const GraphValueId gold_weight = gold.AddWeight(
            WeightSpec(),
            {.slot = ParameterSlot::kScale,
             .decoder_layer_index = 0U,
             .semantic_role = TransformerWeightRole::kInputNorm});
    const auto gold_result = gold.AddNode(
            OpType::kRmsNorm, 0U, {gold_input, gold_weight},
            {{.payload = ActivationValue{}}}, RmsNormParams{});
    ASSERT_TRUE(gold_result.ok());
    const auto& gold_node = gold.GetNode((*gold_result).node);
    const auto& gold_value = gold.GetValue((*gold_result).outputs[0]);

    // Replicate via the test constructor with exact derived metadata.
    std::vector<GraphValue> values = {
            {.payload = ConstantValue{{}},
             .spec = ActivationSpec(),
             .name = "input"},
            {.payload = WeightValue{.binding =
                                            {.slot = ParameterSlot::kScale,
                                             .decoder_layer_index = 0U,
                                             .semantic_role = TransformerWeightRole::kInputNorm}},
             .spec = WeightSpec(),
             .name = "weight"},
            {.payload = ActivationValue{},
             .spec = gold_value.spec,
             .producer = GraphNodeId{0},
             .name = "out"},
    };
    std::vector<GraphNode> nodes = {
            {.op_type = OpType::kRmsNorm,
             .decoder_layer_index = 0U,
             .inputs = {GraphValueId{0}, GraphValueId{1}},
             .outputs = {GraphValueId{2}},
             .op_params = RmsNormParams{},
             .runtime_checks = gold_node.runtime_checks,
             .name = "rms_norm"},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();
    EXPECT_TRUE(status.ok()) << status.ToString();
}

}// namespace
}// namespace aethermind
