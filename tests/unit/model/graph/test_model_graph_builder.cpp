#include "aethermind/model/graph/graph_builder.h"

#include <gtest/gtest.h>

#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace aethermind {
namespace {

struct TestStorage : RawStorage {};

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

const TensorSpec& OnlyOneOutput(const ModelGraph& graph, const GraphNode& node) {
    EXPECT_EQ(node.outputs.size(), 1U);
    return graph.GetValue(node.outputs.front()).spec;
}

const WeightBinding& WeightBindingAt(const ModelGraph& graph, const GraphNode& node, size_t input_index) {
    const GraphValue& value = graph.GetValue(node.inputs[input_index]);
    return std::get<WeightValue>(value.payload).binding;
}

const StateBinding& StateBindingForValue(const ModelGraph& graph, GraphValueId value_id) {
    const GraphValue& value = graph.GetValue(value_id);
    return std::get<StateValue>(value.payload).binding;
}

const KVCacheStateBinding& KVCacheBindingForValue(const ModelGraph& graph, GraphValueId value_id) {
    return std::get<KVCacheStateBinding>(StateBindingForValue(graph, value_id));
}

const GraphNode* FindNodeByDebugName(const ModelGraph& graph, std::string_view debug_name) {
    for (const GraphNode& node: graph.GetNodes()) {
        if (std::string_view(node.debug_name) == debug_name) {
            return &node;
        }
    }
    return nullptr;
}

void ExpectLayerWeightBinding(const ModelGraph& graph,
                              const GraphNode& node,
                              WeightRole role,
                              uint32_t layer_index) {
    const WeightBinding& binding = WeightBindingAt(graph, node, 1);
    EXPECT_EQ(binding.role, role);
    ASSERT_TRUE(binding.decoder_layer_index.has_value());
    EXPECT_EQ(*binding.decoder_layer_index, layer_index);
}

TEST(ModelGraphBuilder, BuildsFullLlamaDenseTopology) {
    const HfModelConfig config = MakeLlamaConfig(2);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    ASSERT_EQ(graph->GetNodes().size(), 1U + 2U * 15U + 3U);
    ASSERT_EQ(graph->GetInputs().size(), 2U);
    EXPECT_EQ(graph->GetInputs()[0].name, "token_ids");
    EXPECT_EQ(graph->GetInputs()[1].name, "position_ids");
    ASSERT_EQ(graph->GetOutputs().size(), 1U);
    EXPECT_EQ(graph->GetOutputs()[0].name, "output_token_ids");
    EXPECT_TRUE(graph->Validate().ok());
    const StatusOr<std::vector<GraphNodeId>> order = graph->TopologicalOrder();
    ASSERT_TRUE(order.ok()) << order.status().ToString();
    ASSERT_EQ(order->size(), graph->GetNodes().size());
    EXPECT_EQ(graph->GetConfig().hidden_size, config.hidden_size);

    const auto nodes = graph->GetNodes();
    EXPECT_EQ(nodes[0].op_type, OpType::kEmbedding);

    constexpr OpType kLayerOps[] = {
            OpType::kRmsNorm,
            OpType::kLinear,
            OpType::kLinear,
            OpType::kLinear,
            OpType::kRoPE,
            OpType::kKVCacheUpdate,
            OpType::kAttention,
            OpType::kLinear,
            OpType::kAdd,
            OpType::kRmsNorm,
            OpType::kLinear,
            OpType::kLinear,
            OpType::kSiluMul,
            OpType::kLinear,
            OpType::kAdd,
    };

    for (size_t layer = 0; layer < 2; ++layer) {
        const size_t offset = 1 + layer * std::size(kLayerOps);
        for (size_t i = 0; i < std::size(kLayerOps); ++i) {
            EXPECT_EQ(nodes[offset + i].op_type, kLayerOps[i]) << "layer=" << layer << ", node=" << i;
            ASSERT_TRUE(nodes[offset + i].decoder_layer_index.has_value());
            EXPECT_EQ(*nodes[offset + i].decoder_layer_index, layer);
        }
    }

    EXPECT_FALSE(nodes[0].decoder_layer_index.has_value());

    const size_t tail = 1 + 2U * 15U;
    EXPECT_EQ(nodes[tail].op_type, OpType::kRmsNorm);
    EXPECT_EQ(nodes[tail + 1].op_type, OpType::kLinear);
    EXPECT_EQ(nodes[tail + 2].op_type, OpType::kArgmax);
    EXPECT_FALSE(nodes[tail].decoder_layer_index.has_value());
    EXPECT_FALSE(nodes[tail + 1].decoder_layer_index.has_value());
    EXPECT_FALSE(nodes[tail + 2].decoder_layer_index.has_value());
}

TEST(ModelGraphBuilder, RecordsWeightBindingsAndRegisteredOperatorParams) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();

    const WeightBinding& token_embedding = WeightBindingAt(*graph, nodes[0], 1);
    EXPECT_EQ(token_embedding.role, WeightRole::kTokenEmbedding);
    EXPECT_FALSE(token_embedding.decoder_layer_index.has_value());
    EXPECT_TRUE(nodes[0].attrs.bytes.empty());
    EXPECT_NE(std::get_if<EmbeddingParams>(&nodes[0].op_params), nullptr);

    const GraphNode& input_norm = nodes[1];
    ExpectLayerWeightBinding(*graph, input_norm, WeightRole::kInputNorm, 0U);
    const auto* rms_params = std::get_if<RmsNormParams>(&input_norm.op_params);
    ASSERT_NE(rms_params, nullptr);
    EXPECT_FLOAT_EQ(rms_params->eps, static_cast<float>(config.rms_norm_eps));

    ExpectLayerWeightBinding(*graph, nodes[2], WeightRole::kAttentionQ, 0U);
    ExpectLayerWeightBinding(*graph, nodes[3], WeightRole::kAttentionK, 0U);
    ExpectLayerWeightBinding(*graph, nodes[4], WeightRole::kAttentionV, 0U);
    ExpectLayerWeightBinding(*graph, nodes[8], WeightRole::kAttentionO, 0U);
    ExpectLayerWeightBinding(*graph, nodes[10], WeightRole::kPostAttentionNorm, 0U);
    ExpectLayerWeightBinding(*graph, nodes[11], WeightRole::kMlpGate, 0U);
    ExpectLayerWeightBinding(*graph, nodes[12], WeightRole::kMlpUp, 0U);
    ExpectLayerWeightBinding(*graph, nodes[14], WeightRole::kMlpDown, 0U);

    const GraphNode& final_norm = nodes[16];
    const WeightBinding& final_norm_weight = WeightBindingAt(*graph, final_norm, 1);
    EXPECT_EQ(final_norm_weight.role, WeightRole::kFinalNorm);
    EXPECT_FALSE(final_norm_weight.decoder_layer_index.has_value());

    const GraphNode& lm_head = nodes[17];
    const WeightBinding& lm_head_weight = WeightBindingAt(*graph, lm_head, 1);
    EXPECT_EQ(lm_head_weight.role, WeightRole::kLmHead);
    EXPECT_FALSE(lm_head_weight.decoder_layer_index.has_value());
}

TEST(ModelGraphBuilder, RecordsTypedParamsForAllGraphOps) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();

    EXPECT_NE(std::get_if<EmbeddingParams>(&nodes[0].op_params), nullptr);
    EXPECT_NE(std::get_if<LinearParams>(&nodes[2].op_params), nullptr);
    EXPECT_NE(std::get_if<LinearParams>(&nodes[3].op_params), nullptr);
    EXPECT_NE(std::get_if<LinearParams>(&nodes[4].op_params), nullptr);

    const auto* rope_params = std::get_if<RoPEParams>(&nodes[5].op_params);
    ASSERT_NE(rope_params, nullptr);
    EXPECT_EQ(rope_params->head_dim, config.head_dim);
    EXPECT_EQ(rope_params->num_attention_heads, config.num_attention_heads);
    EXPECT_EQ(rope_params->num_key_value_heads, config.num_key_value_heads);
    EXPECT_EQ(rope_params->max_position_embeddings, config.max_position_embeddings);
    EXPECT_DOUBLE_EQ(rope_params->theta, config.rope.theta);

    EXPECT_NE(std::get_if<KVCacheUpdateParams>(&nodes[6].op_params), nullptr);

    const auto* attention_params = std::get_if<AttentionParams>(&nodes[7].op_params);
    ASSERT_NE(attention_params, nullptr);
    EXPECT_EQ(attention_params->head_dim, config.head_dim);
    EXPECT_EQ(attention_params->num_attention_heads, config.num_attention_heads);
    EXPECT_EQ(attention_params->num_key_value_heads, config.num_key_value_heads);

    EXPECT_NE(std::get_if<AddParams>(&nodes[9].op_params), nullptr);
    EXPECT_NE(std::get_if<SiluMulParams>(&nodes[13].op_params), nullptr);
    EXPECT_NE(std::get_if<AddParams>(&nodes[15].op_params), nullptr);
    EXPECT_NE(std::get_if<LinearParams>(&nodes[17].op_params), nullptr);
    const auto* argmax_params = std::get_if<ArgmaxParams>(&nodes[18].op_params);
    ASSERT_NE(argmax_params, nullptr);
    EXPECT_EQ(argmax_params->axis, -1);
}

TEST(ModelGraphBuilder, TracesResidualDataflowInAttention) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();
    const GraphNode& embedding = nodes[0];
    const GraphNode& attention_o = nodes[8];
    const GraphNode& residual_add = nodes[9];
    ASSERT_EQ(residual_add.inputs.size(), 2U);
    EXPECT_EQ(residual_add.inputs[0], embedding.outputs[0]);
    EXPECT_EQ(residual_add.inputs[1], attention_o.outputs[0]);

    const StatusOr<std::vector<std::vector<GraphNodeId>>> index = BuildConsumerIndex(*graph);
    ASSERT_TRUE(index.ok()) << index.status().ToString();
    const std::span<const GraphNodeId> hidden_consumers = GetConsumers(*index, embedding.outputs[0]);
    ASSERT_EQ(hidden_consumers.size(), 2U);
    EXPECT_EQ(hidden_consumers[0], GraphNodeId{1});
    EXPECT_EQ(hidden_consumers[1], GraphNodeId{9});
}

TEST(ModelGraphBuilder, TracesRopeDualOutputDataflow) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();
    const GraphNode& rope = nodes[5];
    const GraphNode& kv_cache_update = nodes[6];
    const GraphNode& attention = nodes[7];
    ASSERT_EQ(rope.outputs.size(), 2U);
    ASSERT_EQ(rope.inputs.size(), 3U);
    EXPECT_EQ(rope.inputs[2], graph->GetInputs()[1].value);
    ASSERT_EQ(kv_cache_update.inputs.size(), 4U);
    EXPECT_EQ(kv_cache_update.inputs[0], rope.outputs[1]);
    ASSERT_EQ(attention.inputs.size(), 3U);
    EXPECT_EQ(attention.inputs[0], rope.outputs[0]);

    const StatusOr<std::vector<std::vector<GraphNodeId>>> index = BuildConsumerIndex(*graph);
    ASSERT_TRUE(index.ok()) << index.status().ToString();
    ASSERT_EQ(GetConsumers(*index, rope.outputs[0]).size(), 1U);
    EXPECT_EQ(GetConsumers(*index, rope.outputs[0])[0], GraphNodeId{7});
    ASSERT_EQ(GetConsumers(*index, rope.outputs[1]).size(), 1U);
    EXPECT_EQ(GetConsumers(*index, rope.outputs[1])[0], GraphNodeId{6});

    const TensorSpec& q_rope = graph->GetValue(rope.outputs[0]).spec;
    const TensorSpec& k_rope = graph->GetValue(rope.outputs[1]).spec;
    EXPECT_EQ(q_rope.shape[1].GetStaticValue(), config.hidden_size);
    EXPECT_EQ(k_rope.shape[1].GetStaticValue(), config.num_key_value_heads * config.head_dim);
}

TEST(ModelGraphBuilder, TracesKvCacheStateDataflow) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();
    const GraphNode& kv_cache_update = nodes[6];
    const GraphNode& attention = nodes[7];

    ASSERT_EQ(kv_cache_update.outputs.size(), 2U);
    ASSERT_EQ(attention.inputs.size(), 3U);
    EXPECT_EQ(attention.inputs[1], kv_cache_update.outputs[0]);
    EXPECT_EQ(attention.inputs[2], kv_cache_update.outputs[1]);

    const GraphValue& k_cache_out = graph->GetValue(kv_cache_update.outputs[0]);
    ASSERT_TRUE(std::holds_alternative<StateValue>(k_cache_out.payload));
    const auto& k_cache_binding = KVCacheBindingForValue(*graph, kv_cache_update.outputs[0]);
    EXPECT_EQ(k_cache_binding.slot, KVCacheSlot::kKey);
    EXPECT_EQ(k_cache_binding.decoder_layer_index, 0U);
    ASSERT_TRUE(k_cache_out.producer.has_value());
    EXPECT_EQ(k_cache_out.producer->index, 6U);

    const GraphValue& v_cache_out = graph->GetValue(kv_cache_update.outputs[1]);
    ASSERT_TRUE(std::holds_alternative<StateValue>(v_cache_out.payload));
    const auto& v_cache_binding = KVCacheBindingForValue(*graph, kv_cache_update.outputs[1]);
    EXPECT_EQ(v_cache_binding.slot, KVCacheSlot::kValue);
    EXPECT_EQ(v_cache_binding.decoder_layer_index, 0U);
    ASSERT_TRUE(v_cache_out.producer.has_value());
    EXPECT_EQ(v_cache_out.producer->index, 6U);

    const GraphValue& consumed_k_cache = graph->GetValue(attention.inputs[1]);
    ASSERT_TRUE(std::holds_alternative<StateValue>(consumed_k_cache.payload));
    EXPECT_EQ(KVCacheBindingForValue(*graph, attention.inputs[1]).slot, KVCacheSlot::kKey);

    const GraphValue& consumed_v_cache = graph->GetValue(attention.inputs[2]);
    ASSERT_TRUE(std::holds_alternative<StateValue>(consumed_v_cache.payload));
    EXPECT_EQ(KVCacheBindingForValue(*graph, attention.inputs[2]).slot, KVCacheSlot::kValue);
}

TEST(ModelGraphBuilder, TracesPerLayerKvCacheStateFamilies) {
    const HfModelConfig config = MakeLlamaConfig(2);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();
    const GraphNode& layer0_kv_cache_update = nodes[6];
    const GraphNode& layer0_attention = nodes[7];
    const GraphNode& layer1_kv_cache_update = nodes[21];
    const GraphNode& layer1_attention = nodes[22];

    ASSERT_EQ(layer0_kv_cache_update.outputs.size(), 2U);
    ASSERT_EQ(layer0_kv_cache_update.inputs.size(), 4U);
    ASSERT_EQ(layer0_attention.inputs.size(), 3U);
    ASSERT_EQ(layer1_kv_cache_update.inputs.size(), 4U);
    ASSERT_EQ(layer1_kv_cache_update.outputs.size(), 2U);
    ASSERT_EQ(layer1_attention.inputs.size(), 3U);

    EXPECT_EQ(layer0_attention.inputs[1], layer0_kv_cache_update.outputs[0]);
    EXPECT_EQ(layer0_attention.inputs[2], layer0_kv_cache_update.outputs[1]);
    EXPECT_NE(layer1_kv_cache_update.inputs[2], layer0_kv_cache_update.outputs[0]);
    EXPECT_NE(layer1_kv_cache_update.inputs[3], layer0_kv_cache_update.outputs[1]);
    EXPECT_EQ(layer1_attention.inputs[1], layer1_kv_cache_update.outputs[0]);
    EXPECT_EQ(layer1_attention.inputs[2], layer1_kv_cache_update.outputs[1]);

    const GraphValue& layer0_k_input_state = graph->GetValue(layer0_kv_cache_update.inputs[2]);
    const GraphValue& layer0_v_input_state = graph->GetValue(layer0_kv_cache_update.inputs[3]);
    const GraphValue& layer1_k_input_state = graph->GetValue(layer1_kv_cache_update.inputs[2]);
    const GraphValue& layer1_v_input_state = graph->GetValue(layer1_kv_cache_update.inputs[3]);
    EXPECT_FALSE(layer0_k_input_state.producer.has_value());
    EXPECT_FALSE(layer0_v_input_state.producer.has_value());
    EXPECT_FALSE(layer1_k_input_state.producer.has_value());
    EXPECT_FALSE(layer1_v_input_state.producer.has_value());

    const KVCacheStateBinding& layer0_k_input_binding = KVCacheBindingForValue(*graph, layer0_kv_cache_update.inputs[2]);
    const KVCacheStateBinding& layer0_k_output_binding = KVCacheBindingForValue(*graph, layer0_kv_cache_update.outputs[0]);
    const KVCacheStateBinding& layer0_v_input_binding = KVCacheBindingForValue(*graph, layer0_kv_cache_update.inputs[3]);
    const KVCacheStateBinding& layer0_v_output_binding = KVCacheBindingForValue(*graph, layer0_kv_cache_update.outputs[1]);
    const KVCacheStateBinding& layer1_k_input_binding = KVCacheBindingForValue(*graph, layer1_kv_cache_update.inputs[2]);
    const KVCacheStateBinding& layer1_k_output_binding = KVCacheBindingForValue(*graph, layer1_kv_cache_update.outputs[0]);
    const KVCacheStateBinding& layer1_v_input_binding = KVCacheBindingForValue(*graph, layer1_kv_cache_update.inputs[3]);
    const KVCacheStateBinding& layer1_v_output_binding = KVCacheBindingForValue(*graph, layer1_kv_cache_update.outputs[1]);

    EXPECT_EQ(layer0_k_input_binding.decoder_layer_index, 0U);
    EXPECT_EQ(layer0_k_output_binding.decoder_layer_index, 0U);
    EXPECT_EQ(layer0_v_input_binding.decoder_layer_index, 0U);
    EXPECT_EQ(layer0_v_output_binding.decoder_layer_index, 0U);
    EXPECT_EQ(layer1_k_input_binding.decoder_layer_index, 1U);
    EXPECT_EQ(layer1_k_output_binding.decoder_layer_index, 1U);
    EXPECT_EQ(layer1_v_input_binding.decoder_layer_index, 1U);
    EXPECT_EQ(layer1_v_output_binding.decoder_layer_index, 1U);

    EXPECT_EQ(layer0_k_input_binding, layer0_k_output_binding);
    EXPECT_EQ(layer0_v_input_binding, layer0_v_output_binding);
    EXPECT_EQ(layer1_k_input_binding, layer1_k_output_binding);
    EXPECT_EQ(layer1_v_input_binding, layer1_v_output_binding);
    EXPECT_EQ(layer0_k_input_binding.slot, KVCacheSlot::kKey);
    EXPECT_EQ(layer0_v_input_binding.slot, KVCacheSlot::kValue);
    EXPECT_EQ(layer1_k_input_binding.slot, KVCacheSlot::kKey);
    EXPECT_EQ(layer1_v_input_binding.slot, KVCacheSlot::kValue);
}

TEST(ModelGraphBuilder, UsesSymbolicSequenceAndStaticModelDimensions) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();

    ASSERT_EQ(graph->GetInputs().size(), 2U);
    const TensorSpec& token_ids = graph->GetValue(graph->GetInputs()[0].value).spec;
    const TensorSpec& position_ids = graph->GetValue(graph->GetInputs()[1].value).spec;
    ASSERT_EQ(nodes[0].inputs.size(), 2U);
    EXPECT_EQ(nodes[0].inputs[0], graph->GetInputs()[0].value);
    ASSERT_EQ(token_ids.shape.rank(), 1U);
    const ShapeSymbol seq_len = token_ids.shape[0];
    EXPECT_TRUE(seq_len.IsSymbolic());
    ASSERT_EQ(position_ids.shape.rank(), 1U);
    EXPECT_EQ(position_ids.shape[0], seq_len);

    const TensorSpec& embedding_output = OnlyOneOutput(*graph, nodes[0]);
    ASSERT_EQ(embedding_output.shape.rank(), 2U);
    EXPECT_EQ(embedding_output.shape[0], seq_len);
    EXPECT_EQ(embedding_output.shape[1].GetStaticValue(), config.hidden_size);

    const GraphNode& attention_o = nodes[8];
    const TensorSpec& attention_o_input = graph->GetValue(attention_o.inputs[0]).spec;
    ASSERT_EQ(attention_o_input.shape.rank(), 2U);
    EXPECT_EQ(attention_o_input.shape[1].GetStaticValue(), config.hidden_size);

    const GraphNode& kv_cache_update = nodes[6];
    ASSERT_EQ(kv_cache_update.inputs.size(), 4U);
    ASSERT_EQ(kv_cache_update.outputs.size(), 2U);
    const TensorSpec& k_cache_in = graph->GetValue(kv_cache_update.inputs[2]).spec;
    const TensorSpec& v_cache_in = graph->GetValue(kv_cache_update.inputs[3]).spec;
    const TensorSpec& k_cache_out = graph->GetValue(kv_cache_update.outputs[0]).spec;
    const TensorSpec& v_cache_out = graph->GetValue(kv_cache_update.outputs[1]).spec;
    ASSERT_EQ(k_cache_in.shape.rank(), 3U);
    ASSERT_EQ(v_cache_in.shape.rank(), 3U);
    ASSERT_EQ(k_cache_out.shape.rank(), 3U);
    ASSERT_EQ(v_cache_out.shape.rank(), 3U);
    EXPECT_EQ(k_cache_in.shape[0].GetStaticValue(), config.num_key_value_heads);
    EXPECT_EQ(v_cache_in.shape[0].GetStaticValue(), config.num_key_value_heads);
    EXPECT_EQ(k_cache_out.shape[0].GetStaticValue(), config.num_key_value_heads);
    EXPECT_EQ(v_cache_out.shape[0].GetStaticValue(), config.num_key_value_heads);
    EXPECT_TRUE(k_cache_in.shape[1].IsSymbolic());
    EXPECT_NE(k_cache_in.shape[1], seq_len);
    EXPECT_EQ(v_cache_in.shape[1], k_cache_in.shape[1]);
    EXPECT_EQ(k_cache_out.shape[1], k_cache_in.shape[1]);
    EXPECT_EQ(v_cache_out.shape[1], k_cache_in.shape[1]);
    EXPECT_EQ(k_cache_in.shape[2].GetStaticValue(), config.head_dim);
    EXPECT_EQ(v_cache_in.shape[2].GetStaticValue(), config.head_dim);
    EXPECT_EQ(k_cache_out.shape[2].GetStaticValue(), config.head_dim);
    EXPECT_EQ(v_cache_out.shape[2].GetStaticValue(), config.head_dim);

    const TensorSpec& logits = OnlyOneOutput(*graph, nodes[17]);
    ASSERT_EQ(logits.shape.rank(), 2U);
    EXPECT_EQ(logits.shape[0], seq_len);
    EXPECT_EQ(logits.shape[1].GetStaticValue(), config.vocab_size);
}

TEST(ModelGraphBuilder, RejectsInvalidConfig) {
    HfModelConfig config = MakeLlamaConfig(1);
    config.hidden_size = 0;
    const ResolvedModelWeights weights = MakeWeights(MakeLlamaConfig(1));

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_FALSE(graph.ok());
    EXPECT_EQ(graph.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraphBuilder, RejectsResolvedLayerCountMismatch) {
    const HfModelConfig config = MakeLlamaConfig(2);
    ResolvedModelWeights weights = MakeWeights(config);
    weights.layers.pop_back();

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_FALSE(graph.ok());
    EXPECT_EQ(graph.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelGraphBuilder, AssignsPyTorchStyleDebugNames) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    constexpr std::string_view kExpectedNames[] = {
            "embed_tokens",
            "layers.0.input_layernorm",
            "layers.0.self_attn.q_proj",
            "layers.0.self_attn.k_proj",
            "layers.0.self_attn.v_proj",
            "layers.0.self_attn.rotary_emb",
            "layers.0.self_attn.kv_cache_update",
            "layers.0.self_attn.attention",
            "layers.0.self_attn.o_proj",
            "layers.0.post_attention_add",
            "layers.0.post_attention_layernorm",
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.up_proj",
            "layers.0.mlp.act",
            "layers.0.mlp.down_proj",
            "layers.0.mlp_add",
            "norm",
            "lm_head",
            "argmax",
    };

    for (std::string_view name: kExpectedNames) {
        EXPECT_NE(FindNodeByDebugName(*graph, name), nullptr) << name;
    }
}

TEST(ModelGraphBuilder, AssignsLayerScopedKvCacheStateDebugNames) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();
    const GraphNode& kv_cache_update = nodes[6];
    ASSERT_EQ(kv_cache_update.inputs.size(), 4U);
    EXPECT_EQ(graph->GetValue(kv_cache_update.inputs[2]).debug_name, "layers.0.self_attn.k_cache");
    EXPECT_EQ(graph->GetValue(kv_cache_update.inputs[3]).debug_name, "layers.0.self_attn.v_cache");
}

TEST(ModelGraphBuilder, TracesMlpResidualDataflow) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();
    const GraphNode& post_attn_add = nodes[9];
    const GraphNode& mlp_down = nodes[14];
    const GraphNode& mlp_add = nodes[15];
    ASSERT_EQ(mlp_add.inputs.size(), 2U);
    EXPECT_EQ(mlp_add.inputs[0], post_attn_add.outputs[0]);
    EXPECT_EQ(mlp_add.inputs[1], mlp_down.outputs[0]);
}

TEST(ModelGraphBuilder, AssignsPyTorchStyleWeightDebugNames) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);
    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();

    EXPECT_EQ(graph->GetValue(nodes[0].inputs[1]).debug_name, "embed_tokens.weight");
    EXPECT_EQ(graph->GetValue(nodes[1].inputs[1]).debug_name, "layers.0.input_layernorm.weight");
    EXPECT_EQ(graph->GetValue(nodes[2].inputs[1]).debug_name, "layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[3].inputs[1]).debug_name, "layers.0.self_attn.k_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[4].inputs[1]).debug_name, "layers.0.self_attn.v_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[8].inputs[1]).debug_name, "layers.0.self_attn.o_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[10].inputs[1]).debug_name, "layers.0.post_attention_layernorm.weight");
    EXPECT_EQ(graph->GetValue(nodes[11].inputs[1]).debug_name, "layers.0.mlp.gate_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[12].inputs[1]).debug_name, "layers.0.mlp.up_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[14].inputs[1]).debug_name, "layers.0.mlp.down_proj.weight");
    EXPECT_EQ(graph->GetValue(nodes[16].inputs[1]).debug_name, "norm.weight");
    EXPECT_EQ(graph->GetValue(nodes[17].inputs[1]).debug_name, "lm_head.weight");
}

}// namespace
}// namespace aethermind
