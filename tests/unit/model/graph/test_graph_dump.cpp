#include "aethermind/model/graph/graph_dump.h"
#include "test_graph_helpers.h"

#include "aethermind/model/graph/graph_builder.h"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace aethermind {
namespace {

struct TestStorage : RawStorage {};

HfModelConfig MakeConfig() {
    return HfModelConfig{.model_type = "llama",
                         .architectures = {"LlamaForCausalLM"},
                         .hidden_size = 8,
                         .intermediate_size = 16,
                         .num_hidden_layers = 1,
                         .num_attention_heads = 4,
                         .num_key_value_heads = 2,
                         .vocab_size = 32,
                         .max_position_embeddings = 128,
                         .head_dim = 2,
                         .rms_norm_eps = 1.0e-5,
                         .hidden_act = "silu",
                         .tie_word_embeddings = false,
                         .weight_dtype_hint = DataType::Float32()};
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

TEST(GraphDump, DumpsMinimalGraph) {
    ModelGraph graph;
    const GraphValueId tokens = graph.AddInput(Spec(DataType::Int(32), {4}), "tokens");
    const GraphValueId weight = graph.AddWeight(Spec(DataType::Float32(), {32, 8}),
                                                WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                              .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                                "tok_embeddings.weight");
    auto embedding_or = graph.AddNode(
            OpType::kEmbedding,
            std::nullopt,
            {tokens, weight},
            {NodeOutputDesc{.payload = ActivationValue{},
                            .name = "hidden"}},
            EmbeddingParams{},
            {},
            "embedding");
    ASSERT_TRUE(embedding_or.ok()) << embedding_or.status().ToString();
    const AddedNode& embedding = *embedding_or;
    graph.MarkOutput(embedding.outputs[0], "hidden_out");

    std::ostringstream os;
    DumpGraph(graph, os);

    const std::string dump = os.str();
    EXPECT_NE(dump.find("ModelGraph"), std::string::npos);
    EXPECT_NE(dump.find("op=Embedding"), std::string::npos);
    EXPECT_NE(dump.find("kind=weight"), std::string::npos);
    EXPECT_NE(dump.find("slot=EmbeddingTable, semantic=TokenEmbedding"), std::string::npos);
    EXPECT_NE(dump.find("EmbeddingParams{}"), std::string::npos);
    EXPECT_NE(dump.find("debug_name=embedding"), std::string::npos);
}

// --- ConstantValue dump coverage ---

TEST(GraphDump, DumpsConstantValue) {
    ModelGraph graph;
    auto inline_data = std::make_shared<const std::vector<std::byte>>(
            std::vector<std::byte>{std::byte{0xAA}, std::byte{0xBB}});
    const GraphValueId constant = graph.AddConstant(
            Spec(DataType::Float32(), {4}),
            ConstantBinding{.inline_data = std::move(inline_data), .name = "rope.sin_cos_table"},
            "rope_table");
    (void) constant;

    std::ostringstream os;
    DumpGraph(graph, os);

    const std::string dump = os.str();
    EXPECT_NE(dump.find("kind=constant"), std::string::npos);
    EXPECT_NE(dump.find("constant(name=rope.sin_cos_table, inline_data=2B)"), std::string::npos);
    EXPECT_NE(dump.find("debug_name=rope_table"), std::string::npos);
}

TEST(GraphDump, DumpsQuantizationWhenSet) {
    ModelGraph graph;
    const GraphValueId weight = graph.AddWeight(
            Spec(DataType::Float32(), {16, 4}),
            WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                          .semantic_role = TransformerWeightRole::kTokenEmbedding},
            "embed.weight");
    graph.SetQuantization(weight, QuantizationSpec{
                                          .kind = QuantizationKind::kInt4,
                                          .group_size = 64,
                                          .scale_dtype = DataType::Float32(),
                                          .has_zero_point = true,
                                  });

    std::ostringstream os;
    DumpGraph(graph, os);

    const std::string dump = os.str();
    EXPECT_NE(dump.find("quant=int4"), std::string::npos);
    EXPECT_NE(dump.find("group_size=64"), std::string::npos);
}

// --- Issue Q: DumpGraph on empty graph ---

TEST(GraphDump, DumpsEmptyGraphWithoutCrash) {
    ModelGraph graph;

    std::ostringstream os;
    DumpGraph(graph, os);

    const std::string dump = os.str();
    EXPECT_NE(dump.find("ModelGraph"), std::string::npos);
    // Empty graph has no inputs, values, or nodes
    EXPECT_EQ(dump.find("op="), std::string::npos);
}

TEST(GraphDump, DumpsLlamaDenseGraph) {
    const HfModelConfig config = MakeConfig();
    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, MakeWeights(config));
    ASSERT_TRUE(graph.ok()) << graph.status().ToString();

    std::ostringstream os;
    DumpGraph(*graph, os);

    const std::string dump = os.str();
    EXPECT_NE(dump.find("op=RmsNorm"), std::string::npos);
    EXPECT_NE(dump.find("op=KVCacheUpdate"), std::string::npos);
    EXPECT_NE(dump.find("state(kv_cache"), std::string::npos);
    EXPECT_NE(dump.find("slot=Kernel, semantic=AttentionQ"), std::string::npos);
    EXPECT_NE(dump.find("slot=Scale, semantic=InputNorm"), std::string::npos);
    EXPECT_NE(dump.find("RoPEParams{"), std::string::npos);
}

TEST(GraphDump, DumpsParameterSlotsAndSemanticRoles) {
    EXPECT_STREQ(ToString(ParameterSlot::kKernel), "Kernel");
    EXPECT_STREQ(ToString(ParameterSlot::kBias), "Bias");
    EXPECT_STREQ(ToString(ParameterSlot::kScale), "Scale");
    EXPECT_STREQ(ToString(ParameterSlot::kShift), "Shift");
    EXPECT_STREQ(ToString(ParameterSlot::kEmbeddingTable), "EmbeddingTable");

    EXPECT_STREQ(ToString(TransformerWeightRole::kTokenEmbedding), "TokenEmbedding");
    EXPECT_STREQ(ToString(TransformerWeightRole::kInputNorm), "InputNorm");
    EXPECT_STREQ(ToString(TransformerWeightRole::kAttentionQ), "AttentionQ");
    EXPECT_STREQ(ToString(TransformerWeightRole::kAttentionK), "AttentionK");
    EXPECT_STREQ(ToString(TransformerWeightRole::kAttentionV), "AttentionV");
    EXPECT_STREQ(ToString(TransformerWeightRole::kAttentionO), "AttentionO");
    EXPECT_STREQ(ToString(TransformerWeightRole::kMlpGate), "MlpGate");
    EXPECT_STREQ(ToString(TransformerWeightRole::kMlpUp), "MlpUp");
    EXPECT_STREQ(ToString(TransformerWeightRole::kMlpDown), "MlpDown");
    EXPECT_STREQ(ToString(TransformerWeightRole::kPostAttentionNorm), "PostAttentionNorm");
    EXPECT_STREQ(ToString(TransformerWeightRole::kFinalNorm), "FinalNorm");
    EXPECT_STREQ(ToString(TransformerWeightRole::kLmHead), "LmHead");
    EXPECT_STREQ(ToString(TransformerWeightRole::kMoERouter), "MoERouter");

    EXPECT_STREQ(ToString(ModelSemanticRole{}), "<none>");
    EXPECT_STREQ(ToString(ModelSemanticRole{TransformerWeightRole::kLmHead}), "LmHead");
}

TEST(GraphDump, DumpsEveryOpParamsVariant) {
    const std::vector<OpParams> params{
            std::monostate{},
            EmbeddingParams{},
            RmsNormParams{.eps = 1.0e-6F},
            LinearParams{},
            RoPEParams{.head_dim = 8,
                       .num_attention_heads = 4,
                       .num_key_value_heads = 2,
                       .max_position_embeddings = 128,
                       .theta = 10000.0,
                       .scaling_factor = 2.0,
                       .scaling_type = HfRopeScalingType::kLinear},
            MatMulParams{.transpose_rhs = true},
            SoftmaxParams{.axis = -1},
            AddParams{},
            SiluMulParams{},
            KVCacheUpdateParams{},
            AttentionParams{.num_attention_heads = 4, .num_key_value_heads = 2, .head_dim = 8},
            ArgmaxParams{.axis = -1},
    };

    std::ostringstream os;
    for (const OpParams& param: params) {
        DumpOpParams(param, os);
        os << '\n';
    }

    const std::string dump = os.str();
    EXPECT_NE(dump.find("monostate{}"), std::string::npos);
    EXPECT_NE(dump.find("EmbeddingParams{}"), std::string::npos);
    EXPECT_NE(dump.find("RmsNormParams{eps="), std::string::npos);
    EXPECT_NE(dump.find("RoPEParams{head_dim=8"), std::string::npos);
    EXPECT_NE(dump.find("MatMulParams{transpose_rhs=true}"), std::string::npos);
    EXPECT_NE(dump.find("AttentionParams{num_attention_heads=4"), std::string::npos);
    EXPECT_NE(dump.find("ArgmaxParams{axis=-1}"), std::string::npos);
}

}// namespace
}// namespace aethermind
