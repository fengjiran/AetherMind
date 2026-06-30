#include "aethermind/model/graph/graph.h"

#include <gtest/gtest.h>

#include <optional>
#include <span>
#include <string>
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

GraphValueId AddEmbeddingOutput(ModelGraph& graph, std::string input_name) {
    const GraphValueId tokens = graph.AddInput(TokenSpec(), std::move(input_name));
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.role = WeightRole::kTokenEmbedding});
    return graph.AddNode(
                        OpType::kEmbedding,
                        std::nullopt,
                        {tokens, weight},
                        {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
                        EmbeddingParams{})
            .outputs.front();
}

ModelGraph BuildValidEmbeddingGraph() {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.role = WeightRole::kTokenEmbedding});
    const AddedNode embedding = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            EmbeddingParams{});
    graph.MarkOutput(embedding.outputs[0], "hidden");
    return graph;
}

TEST(ModelGraph, PublicApiCreatesInputsWeightsNodesAndOutputs) {
    ModelGraph graph;

    const GraphValueId input = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), WeightBinding{.role = WeightRole::kInputNorm});
    const AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}, .debug_name = "normed"}},
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
    const GraphValueId weight = graph.AddWeight(WeightSpec(), WeightBinding{.role = WeightRole::kInputNorm});

    const AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RmsNormParams{.eps = 2.5e-3F});

    const auto* params = std::get_if<RmsNormParams>(&graph.GetNode(node.node).op_params);
    ASSERT_NE(params, nullptr);
    EXPECT_FLOAT_EQ(params->eps, 2.5e-3F);
}

TEST(ModelGraph, ValidateRejectsMissingOpParams) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.role = WeightRole::kTokenEmbedding});
    [[maybe_unused]] const AddedNode node = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsWrongOpParamsForOpType) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(TokenSpec(), "token_ids");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}), WeightBinding{.role = WeightRole::kTokenEmbedding});
    [[maybe_unused]] const AddedNode node = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RmsNormParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidRmsNormEps) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), WeightBinding{.role = WeightRole::kInputNorm});
    [[maybe_unused]] const AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            RmsNormParams{.eps = 0.0F});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsInvalidRoPEParams) {
    ModelGraph graph;
    const GraphValueId q = graph.AddInput(ActivationSpec(), "q");
    const GraphValueId k = graph.AddInput(ActivationSpec(), "k");
    const GraphValueId position_ids = graph.AddInput(ActivationSpec(), "position_ids");
    [[maybe_unused]] const AddedNode node = graph.AddNode(
            OpType::kRoPE,
            0U,
            {q, k, position_ids},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
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
    ModelGraph graph({}, {}, {GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .debug_name = ""}});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsExternalValueWithProducer) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
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
    const auto* binding = std::get_if<KVCacheStateBinding>(&std::get<StateValue>(value.payload).binding);
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
            ConstantBinding{.name = "rope.sin_cos_table", .inline_data = std::move(inline_data)},
            "rope_table");

    ASSERT_EQ(graph.GetValues().size(), 1U);
    const GraphValue& value = graph.GetValue(constant);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(value.payload));
    const ConstantBinding& binding = std::get<ConstantValue>(value.payload).binding;
    EXPECT_EQ(binding.name, "rope.sin_cos_table");
    ASSERT_TRUE(binding.inline_data != nullptr);
    EXPECT_EQ(binding.inline_data->size(), 3U);
    EXPECT_FALSE(value.producer.has_value());
    EXPECT_EQ(value.debug_name, "rope_table");
}

TEST(ModelGraph, SetQuantizationAttachesSchemeToValue) {
    ModelGraph graph;
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            WeightBinding{.role = WeightRole::kTokenEmbedding},
            "embed.weight");

    graph.SetQuantization(weight, QuantizationSpec{
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
            WeightSpec(), WeightBinding{.role = WeightRole::kTokenEmbedding});
    const GraphValueId bias = graph.AddConstant(
            Spec(DataType::Float32(), {8}),
            ConstantBinding{.name = "embed.bias"});
    (void) bias;
    const AddedNode embedding = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            EmbeddingParams{});
    graph.MarkOutput(embedding.outputs[0], "hidden");

    EXPECT_TRUE(graph.Validate().ok());
}

TEST(ModelGraph, ValidateRejectsConstantValueWithProducer) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ConstantValue{.binding = ConstantBinding{.name = "c"}},
                       .spec = Spec(DataType::Float32(), {1}),
                       .producer = GraphNodeId{0}},
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
            WeightBinding{.role = WeightRole::kTokenEmbedding});
    const GraphValueId hidden = graph.AddNode(
                                             OpType::kEmbedding,
                                             std::nullopt,
                                             {tokens, embedding_weight},
                                             {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
                                             EmbeddingParams{})
                                        .outputs.front();
    const GraphValueId k_weight = graph.AddWeight(WeightSpec(), WeightBinding{.role = WeightRole::kAttentionK});
    const GraphValueId v_weight = graph.AddWeight(WeightSpec(), WeightBinding{.role = WeightRole::kAttentionV});
    const GraphValueId k = graph.AddNode(
                                        OpType::kLinear,
                                        std::nullopt,
                                        {hidden, k_weight},
                                        {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
                                        LinearParams{})
                                   .outputs.front();
    const GraphValueId v = graph.AddNode(
                                        OpType::kLinear,
                                        std::nullopt,
                                        {hidden, v_weight},
                                        {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
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

    const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = KStateBinding(0U)}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding(0U)}}},
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
    const GraphValueId layer0_k_cache = graph.AddState(ActivationSpec(), KStateBinding(0U), "layer0_k_cache");
    const GraphValueId layer0_v_cache = graph.AddState(ActivationSpec(), VStateBinding(0U), "layer0_v_cache");

    [[maybe_unused]] const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            1U,
            {k, v, layer0_k_cache, layer0_v_cache},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = KStateBinding(1U)}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding(1U)}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateKStateFamilyMismatch) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(ActivationSpec(), KStateBinding(0U), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(ActivationSpec(), VStateBinding(0U), "v_cache_in");

    [[maybe_unused]] const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = KStateBinding(1U)}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding(0U)}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateVStateFamilyMismatch) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(ActivationSpec(), KStateBinding(0U), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(ActivationSpec(), VStateBinding(0U), "v_cache_in");

    [[maybe_unused]] const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = KStateBinding(0U)}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding(1U)}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateWithSwappedKeyValueStateSlots) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId wrong_k_cache_in = graph.AddState(ActivationSpec(), VStateBinding(0U), "wrong_k_cache_in");
    const GraphValueId wrong_v_cache_in = graph.AddState(ActivationSpec(), KStateBinding(0U), "wrong_v_cache_in");

    [[maybe_unused]] const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, wrong_k_cache_in, wrong_v_cache_in},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding(0U)}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = KStateBinding(0U)}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateWithMixedKeyValueStateLayers) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(ActivationSpec(), KStateBinding(0U), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(ActivationSpec(), VStateBinding(1U), "v_cache_in");

    [[maybe_unused]] const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = KStateBinding(0U)}},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding(1U)}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsAttentionWithSwappedKeyValueStateSlots) {
    ModelGraph graph;
    const GraphValueId q = AddEmbeddingOutput(graph, "tokens_q");
    const GraphValueId wrong_k_cache = graph.AddState(ActivationSpec(), VStateBinding(0U), "wrong_k_cache");
    const GraphValueId wrong_v_cache = graph.AddState(ActivationSpec(), KStateBinding(0U), "wrong_v_cache");

    [[maybe_unused]] const AddedNode attention = graph.AddNode(
            OpType::kAttention,
            0U,
            {q, wrong_k_cache, wrong_v_cache},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            AttentionParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsAttentionWithWrongCacheLayer) {
    ModelGraph graph;
    const GraphValueId q = AddEmbeddingOutput(graph, "tokens_q");
    const GraphValueId k_cache = graph.AddState(ActivationSpec(), KStateBinding(1U), "k_cache");
    const GraphValueId v_cache = graph.AddState(ActivationSpec(), VStateBinding(1U), "v_cache");

    [[maybe_unused]] const AddedNode attention = graph.AddNode(
            OpType::kAttention,
            0U,
            {q, k_cache, v_cache},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}},
            AttentionParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsStateValueWithInvalidProducer) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = StateValue{.binding = KvStateBinding()}, .spec = ActivationSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
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
    const GraphValueId state = graph.AddState(ActivationSpec(), KvStateBinding(), "kv_cache");
    graph.MarkOutput(state, "bad_state_output");

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsKVCacheUpdateWithActivationOutput) {
    ModelGraph graph;
    const GraphValueId k = AddEmbeddingOutput(graph, "tokens_k");
    const GraphValueId v = AddEmbeddingOutput(graph, "tokens_v");
    const GraphValueId k_cache_in = graph.AddState(ActivationSpec(), KStateBinding(), "k_cache_in");
    const GraphValueId v_cache_in = graph.AddState(ActivationSpec(), VStateBinding(), "v_cache_in");

    [[maybe_unused]] const AddedNode update = graph.AddNode(
            OpType::kKVCacheUpdate,
            0U,
            {k, v, k_cache_in, v_cache_in},
            {NodeOutputDesc{.spec = ActivationSpec()},
             NodeOutputDesc{.spec = ActivationSpec(), .payload = StateValue{.binding = VStateBinding()}}},
            KVCacheUpdateParams{});

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsNodeOutputReusingInputValue) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec(), .debug_name = ""},
            GraphValue{.payload = WeightValue{.binding = WeightBinding{.role = WeightRole::kTokenEmbedding}}, .spec = Spec(DataType::Float32(), {32, 8}), .debug_name = ""},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{1}, .debug_name = ""},
            GraphValue{.payload = StateValue{.binding = KStateBinding()}, .spec = ActivationSpec(), .producer = GraphNodeId{2}, .debug_name = ""},
            GraphValue{.payload = StateValue{.binding = VStateBinding()}, .spec = ActivationSpec(), .producer = GraphNodeId{2}, .debug_name = ""},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kEmbedding, .inputs = {GraphValueId{0}, GraphValueId{1}}, .outputs = {GraphValueId{2}}, .op_params = EmbeddingParams{}, .debug_name = ""},
            GraphNode{.op_type = OpType::kEmbedding, .inputs = {GraphValueId{0}, GraphValueId{1}}, .outputs = {GraphValueId{3}}, .op_params = EmbeddingParams{}, .debug_name = ""},
            GraphNode{.op_type = OpType::kKVCacheUpdate, .inputs = {GraphValueId{2}, GraphValueId{3}, GraphValueId{4}, GraphValueId{5}}, .outputs = {GraphValueId{4}, GraphValueId{5}}, .op_params = KVCacheUpdateParams{}, .debug_name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsSchemaPortKindMismatch) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec(), .debug_name = ""},
            GraphValue{.payload = ActivationValue{}, .spec = WeightSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kEmbedding, .inputs = {GraphValueId{0}, GraphValueId{1}}, .outputs = {GraphValueId{2}}, .debug_name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsSchemaArityMismatch) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ModelInputValue{}, .spec = TokenSpec(), .debug_name = ""},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kEmbedding, .inputs = {GraphValueId{0}}, .outputs = {GraphValueId{1}}, .debug_name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, ValidateRejectsOutputOnExternalValue) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const GraphValueId weight = graph.AddWeight(WeightSpec(), WeightBinding{.role = WeightRole::kInputNorm});
    const AddedNode node = graph.AddNode(
            OpType::kRmsNorm,
            0U,
            {input, weight},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}, .debug_name = ""}});
    graph.MarkOutput(input, "bad_output");

    const Status status = graph.Validate();

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, BuildConsumerIndexMapsConsumers) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const AddedNode softmax = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}, .debug_name = ""}});
    const AddedNode argmax = graph.AddNode(
            OpType::kArgmax,
            std::nullopt,
            {softmax.outputs[0]},
            {NodeOutputDesc{.spec = TokenSpec(), .payload = ActivationValue{}, .debug_name = ""}});
    const AddedNode another = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {softmax.outputs[0]},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}, .debug_name = ""}});

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
    const AddedNode softmax = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}, .debug_name = ""}});
    const AddedNode argmax = graph.AddNode(
            OpType::kArgmax,
            std::nullopt,
            {softmax.outputs[0]},
            {NodeOutputDesc{.spec = TokenSpec(), .payload = ActivationValue{}, .debug_name = ""}});

    const StatusOr<std::vector<GraphNodeId>> order = graph.TopologicalOrder();

    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), 2U);
    EXPECT_EQ((*order)[0], softmax.node);
    EXPECT_EQ((*order)[1], argmax.node);
}

TEST(ModelGraph, TopologicalOrderRejectsActivationCycle) {
    std::vector<GraphValue> values = {
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{0}, .debug_name = ""},
            GraphValue{.payload = ActivationValue{}, .spec = ActivationSpec(), .producer = GraphNodeId{1}, .debug_name = ""},
    };
    std::vector<GraphNode> nodes = {
            GraphNode{.op_type = OpType::kSoftmax, .inputs = {GraphValueId{1}}, .outputs = {GraphValueId{0}}, .debug_name = ""},
            GraphNode{.op_type = OpType::kSoftmax, .inputs = {GraphValueId{0}}, .outputs = {GraphValueId{1}}, .debug_name = ""},
    };
    ModelGraph graph({}, std::move(nodes), std::move(values));

    const StatusOr<std::vector<GraphNodeId>> order = graph.TopologicalOrder();

    ASSERT_FALSE(order.ok());
    EXPECT_EQ(order.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraph, FindNodesByOpTypeReturnsMatchingIdsInIndexOrder) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    const AddedNode softmax_a = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}});
    const AddedNode argmax = graph.AddNode(
            OpType::kArgmax,
            std::nullopt,
            {softmax_a.outputs[0]},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}});
    const AddedNode softmax_b = graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {argmax.outputs[0]},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}});

    const std::vector<GraphNodeId> softmax_ids = graph.FindNodesByOpType(OpType::kSoftmax);

    ASSERT_EQ(softmax_ids.size(), 2U);
    EXPECT_EQ(softmax_ids[0], softmax_a.node);
    EXPECT_EQ(softmax_ids[1], softmax_b.node);
    EXPECT_EQ(graph.GetNode(softmax_ids[0]).op_type, OpType::kSoftmax);
}

TEST(ModelGraph, FindNodesByOpTypeReturnsEmptyWhenNoMatch) {
    ModelGraph graph;
    const GraphValueId input = graph.AddInput(ActivationSpec(), "input");
    (void) graph.AddNode(
            OpType::kSoftmax,
            std::nullopt,
            {input},
            {NodeOutputDesc{.spec = ActivationSpec(), .payload = ActivationValue{}}});

    const std::vector<GraphNodeId> ids = graph.FindNodesByOpType(OpType::kAttention);

    EXPECT_TRUE(ids.empty());
}

TEST(ModelGraph, FindNodesByOpTypeReturnsEmptyForEmptyGraph) {
    ModelGraph graph;

    const std::vector<GraphNodeId> ids = graph.FindNodesByOpType(OpType::kSoftmax);

    EXPECT_TRUE(ids.empty());
}

}// namespace
}// namespace aethermind
