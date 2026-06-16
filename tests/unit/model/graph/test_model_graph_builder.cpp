#include "aethermind/model/graph/model_graph_builder.h"

#include "aethermind/operators/embedding_op.h"
#include "aethermind/operators/rmsnorm_op.h"

#include <gtest/gtest.h>

#include <any>
#include <memory>
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

const TensorSpec& OnlyOutput(const GraphNode& node) {
    EXPECT_EQ(node.outputs.size(), 1U);
    return node.outputs.front();
}

TEST(ModelGraphBuilder, BuildsFullLlamaDenseTopology) {
    const HfModelConfig config = MakeLlamaConfig(2);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    ASSERT_EQ(graph->GetNodes().size(), 1U + 2U * 16U + 3U);
    EXPECT_EQ(graph->GetConfig().hidden_size, config.hidden_size);

    const auto nodes = graph->GetNodes();
    EXPECT_EQ(nodes[0].op_type, OpType::kEmbedding);

    constexpr OpType kLayerOps[] = {
            OpType::kRmsNorm,
            OpType::kLinear,
            OpType::kLinear,
            OpType::kLinear,
            OpType::kRoPE,
            OpType::kMatMul,
            OpType::kSoftmax,
            OpType::kMatMul,
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

    const size_t tail = 1 + 2U * 16U;
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

    ASSERT_EQ(nodes[0].weights.size(), 1U);
    EXPECT_EQ(nodes[0].weights[0].role, ModelWeightRole::kTokenEmbedding);
    EXPECT_FALSE(nodes[0].weights[0].decoder_layer_index.has_value());
    EXPECT_TRUE(nodes[0].attrs.bytes.empty());
    EXPECT_NE(std::any_cast<EmbeddingOp::Params>(&nodes[0].op_params), nullptr);

    const GraphNode& input_norm = nodes[1];
    ASSERT_EQ(input_norm.weights.size(), 1U);
    EXPECT_EQ(input_norm.weights[0].role, ModelWeightRole::kInputNorm);
    ASSERT_TRUE(input_norm.weights[0].decoder_layer_index.has_value());
    EXPECT_EQ(*input_norm.weights[0].decoder_layer_index, 0U);
    const auto* rms_params = std::any_cast<RmsNormOp::Params>(&input_norm.op_params);
    ASSERT_NE(rms_params, nullptr);
    EXPECT_FLOAT_EQ(rms_params->eps, static_cast<float>(config.rms_norm_eps));

    const GraphNode& q_proj = nodes[2];
    ASSERT_EQ(q_proj.weights.size(), 1U);
    EXPECT_EQ(q_proj.weights[0].role, ModelWeightRole::kAttentionQ);
    ASSERT_TRUE(q_proj.weights[0].decoder_layer_index.has_value());
    EXPECT_EQ(*q_proj.weights[0].decoder_layer_index, 0U);

    const GraphNode& final_norm = nodes[17];
    ASSERT_EQ(final_norm.weights.size(), 1U);
    EXPECT_EQ(final_norm.weights[0].role, ModelWeightRole::kFinalNorm);
    EXPECT_FALSE(final_norm.weights[0].decoder_layer_index.has_value());

    const GraphNode& lm_head = nodes[18];
    ASSERT_EQ(lm_head.weights.size(), 1U);
    EXPECT_EQ(lm_head.weights[0].role, ModelWeightRole::kLmHead);
    EXPECT_FALSE(lm_head.weights[0].decoder_layer_index.has_value());
}

TEST(ModelGraphBuilder, UsesSymbolicSequenceAndStaticModelDimensions) {
    const HfModelConfig config = MakeLlamaConfig(1);
    const ResolvedModelWeights weights = MakeWeights(config);

    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);

    ASSERT_TRUE(graph.ok()) << graph.status().ToString();
    const auto nodes = graph->GetNodes();

    ASSERT_EQ(nodes[0].inputs.size(), 2U);
    ASSERT_EQ(nodes[0].inputs[0].shape.rank(), 1U);
    const ShapeSymbol seq_len = nodes[0].inputs[0].shape[0];
    EXPECT_TRUE(seq_len.IsSymbolic());

    const TensorSpec& embedding_output = OnlyOutput(nodes[0]);
    ASSERT_EQ(embedding_output.shape.rank(), 2U);
    EXPECT_EQ(embedding_output.shape[0], seq_len);
    EXPECT_EQ(embedding_output.shape[1].GetStaticValue(), config.hidden_size);

    const TensorSpec& logits = OnlyOutput(nodes[18]);
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

}// namespace
}// namespace aethermind
