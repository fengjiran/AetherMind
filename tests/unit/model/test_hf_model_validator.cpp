#include "aethermind/model/formats/hf/hf_model_validator.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace aethermind {
namespace {

struct TestStorage : RawStorage {
    std::vector<std::byte> data;
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
};

HfModelConfig MakeValidLlamaConfig() {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 4096,
            .intermediate_size = 11008,
            .num_hidden_layers = 32,
            .num_attention_heads = 32,
            .num_key_value_heads = 8,
            .vocab_size = 32000,
            .max_position_embeddings = 4096,
            .head_dim = 128,
            .rms_norm_eps = 1e-6,
            .tie_word_embeddings = false,
    };
}

void AddWeight(RawWeightTable* weights,
               std::string name,
               const std::shared_ptr<TestStorage>& storage,
               size_t offset,
               std::vector<int64_t> shape = {1}) {
    weights->emplace(std::move(name),
                     RawWeightView{
                             .data = storage->data.data() + offset,
                             .bytes = 4,
                             .dtype = DataType::Float32(),
                             .shape = std::move(shape),
                             .storage = storage,
                     });
}

RawWeightTable MakeCompleteWeightSet(const HfModelConfig& config, bool include_lm_head = true) {
    auto storage = std::make_shared<TestStorage>((config.num_hidden_layers * 9 + 2 + (include_lm_head ? 1 : 0)) * 4);

    RawWeightTable weights;
    size_t offset = 0;
    AddWeight(&weights, "model.embed_tokens.weight", storage, offset, {1, 1});
    offset += 4;
    AddWeight(&weights, "model.norm.weight", storage, offset);
    offset += 4;

    if (include_lm_head) {
        AddWeight(&weights, "lm_head.weight", storage, offset, {1, 1});
        offset += 4;
    }

    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        AddWeight(&weights, prefix + ".self_attn.q_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".self_attn.k_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".self_attn.v_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".self_attn.o_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".mlp.gate_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".mlp.up_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".mlp.down_proj.weight", storage, offset, {1, 1});
        offset += 4;
        AddWeight(&weights, prefix + ".input_layernorm.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".post_attention_layernorm.weight", storage, offset);
        offset += 4;
    }

    return weights;
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsValidLlamaConfig) {
    const HfModelConfig config = MakeValidLlamaConfig();

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsValidLlamaConfigWithOptions) {
    const HfModelConfig config = MakeValidLlamaConfig();
    const ModelValidationOptions options{};

    const Status status = HfModelValidator::ValidateConfig(config, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonLlamaModelTypeEvenWithLlamaArchitecture) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.model_type = "unknown";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsMissingArchitecturesWhenModelTypeIsLlama) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.architectures.clear();

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnsupportedModelFamily) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.model_type = "gpt_neox";
    config.architectures = {"GPTNeoXForCausalLM"};

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonPositiveRequiredDimensions) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonPositiveRmsNormEps) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rms_norm_eps = 0.0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsHiddenSizeNotDivisibleByAttentionHeads) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 4097;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsTooManyKeyValueHeads) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_key_value_heads = 33;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsAttentionHeadsNotDivisibleByKeyValueHeads) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_key_value_heads = 7;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsIntermediateSizeSmallerThanHiddenSize) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.intermediate_size = config.hidden_size - 1;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonPositiveMaxPositionEmbeddings) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.max_position_embeddings = 0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMismatchedHeadDim) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.head_dim = 64;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnsupportedHiddenActivation) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_act = "linear";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsGeluActivation) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_act = "gelu";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsReluActivation) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_act = "relu";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsBiasByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.attention_bias = true;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsBiasWhenAllowed) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.mlp_bias = true;
    ModelValidationOptions options{};
    options.allow_bias = true;

    const Status status = HfModelValidator::ValidateConfig(config, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonPositiveRopeTheta) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rope.theta = 0.0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsRopeScalingByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rope.scaling_factor = 2.0;
    config.rope.scaling_type = "linear";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsPositiveRopeScalingWhenAllowed) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rope.scaling_factor = 2.0;
    config.rope.scaling_type = "linear";
    ModelValidationOptions options{};
    options.allow_rope_scaling = true;

    const Status status = HfModelValidator::ValidateConfig(config, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnsupportedRopeScalingTypeWhenAllowed) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rope.scaling_factor = 2.0;
    config.rope.scaling_type = "unknown";
    ModelValidationOptions options{};
    options.allow_rope_scaling = true;

    const Status status = HfModelValidator::ValidateConfig(config, options);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsPartialRopeScalingTypeByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rope.scaling_type = "linear";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsPartialRopeScalingWhenAllowed) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rope.scaling_type = "linear";
    ModelValidationOptions options{};
    options.allow_rope_scaling = true;

    const Status status = HfModelValidator::ValidateConfig(config, options);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsQuantizedDTypeHint) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.weight_dtype_hint_name = "int8";
    config.weight_dtype_hint = DataType::Int(8);

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnknownNamedDTypeHint) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.weight_dtype_hint_name = "float8_e4m3fn";
    config.weight_dtype_hint = DataType{};

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsCompleteWeightSet) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    const RawWeightTable weights = MakeCompleteWeightSet(config);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsCompleteWeightSetWithOptions) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    const RawWeightTable weights = MakeCompleteWeightSet(config);
    const ModelValidationOptions options{};

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, ValidateResolvedModelApiExistsButIsNotImplemented) {
    const HfModelConfig config = MakeValidLlamaConfig();
    const ModelWeightIndex resolved{};

    const Status status = HfModelValidator::ValidateResolvedModel(config, resolved);

    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingEmbeddingWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.embed_tokens.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.embed_tokens.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingFinalNormWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.norm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.norm.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingLayerAttentionWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.1.self_attn.q_proj.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.1.self_attn.q_proj.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingLayerMlpWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.0.mlp.down_proj.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.mlp.down_proj.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingLayerNormWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.0.post_attention_layernorm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.post_attention_layernorm.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsWeightSetWithInvalidLayerCount) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 0;

    const Status status = HfModelValidator::ValidateWeightSet(config, RawWeightTable{});

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsWeightWithMismatchedByteSize) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    auto storage = std::make_shared<TestStorage>(4);

    RawWeightTable weights;
    weights.emplace("model.embed_tokens.weight",
                    RawWeightView{
                            .data = storage->data.data(),
                            .bytes = 8,// shape [1] × Float32 = 4, not 8
                            .dtype = DataType::Float32(),
                            .shape = {1},
                            .storage = storage,
                    });
    weights.emplace("model.norm.weight",
                    RawWeightView{
                            .data = storage->data.data(),
                            .bytes = 4,
                            .dtype = DataType::Float32(),
                            .shape = {1},
                            .storage = storage,
                    });
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.self_attn.k_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.self_attn.v_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.self_attn.o_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.mlp.gate_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.mlp.up_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.mlp.down_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.input_layernorm.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.post_attention_layernorm.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.embed_tokens.weight"), std::string::npos);
    EXPECT_NE(status.message().find("byte size mismatch"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsWeightWithNonPositiveShapeDimension) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    auto storage = std::make_shared<TestStorage>(4);

    RawWeightTable weights;
    weights.emplace("model.embed_tokens.weight",
                    RawWeightView{
                            .data = storage->data.data(),
                            .bytes = 0,
                            .dtype = DataType::Float32(),
                            .shape = {0},// non-positive
                            .storage = storage,
                    });
    weights.emplace("model.norm.weight",
                    RawWeightView{
                            .data = storage->data.data(),
                            .bytes = 4,
                            .dtype = DataType::Float32(),
                            .shape = {1},
                            .storage = storage,
                    });
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.self_attn.k_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.self_attn.v_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.self_attn.o_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.mlp.gate_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.mlp.up_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.mlp.down_proj.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.input_layernorm.weight", storage, 0);
    AddWeight(&weights, "model.layers.0.post_attention_layernorm.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("non-positive shape dimension"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsWeightWithNullStorage) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;

    RawWeightTable weights;
    weights.emplace("model.embed_tokens.weight",
                    RawWeightView{
                            .data = nullptr,
                            .bytes = 0,
                            .dtype = DataType::Float32(),
                            .shape = {1},
                            .storage = nullptr,
                    });
    weights.emplace("model.norm.weight",
                    RawWeightView{
                            .data = nullptr,
                            .bytes = 0,
                            .dtype = DataType::Float32(),
                            .shape = {1},
                            .storage = nullptr,
                    });
    const auto add_null = [&](std::string name) {
        weights.emplace(std::move(name),
                        RawWeightView{
                                .data = nullptr,
                                .bytes = 0,
                                .dtype = DataType::Float32(),
                                .shape = {1},
                                .storage = nullptr,
                        });
    };
    add_null("model.layers.0.self_attn.q_proj.weight");
    add_null("model.layers.0.self_attn.k_proj.weight");
    add_null("model.layers.0.self_attn.v_proj.weight");
    add_null("model.layers.0.self_attn.o_proj.weight");
    add_null("model.layers.0.mlp.gate_proj.weight");
    add_null("model.layers.0.mlp.up_proj.weight");
    add_null("model.layers.0.mlp.down_proj.weight");
    add_null("model.layers.0.input_layernorm.weight");
    add_null("model.layers.0.post_attention_layernorm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("invalid RawWeightView"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsEmptyWeightTensor) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    auto storage = std::make_shared<TestStorage>(4);
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.at("model.layers.0.self_attn.q_proj.weight") = RawWeightView{
            .data = storage->data.data(),
            .bytes = 0,
            .dtype = DataType::Float32(),
            .shape = {1, 1},
            .storage = storage,
    };

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("empty tensor"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonContiguousWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.at("model.layers.0.self_attn.q_proj.weight").is_contiguous = false;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("non-contiguous"), std::string::npos);
}

// --- P0-1: Unsupported tensor detection (§8.4) ---

TEST(ModelLoader_HfModelValidatorTest, RejectsMoeTensor) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.mlp.experts.0.up_proj.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("experts"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsRouterTensor) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.mlp.router.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("router"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsBiasTensorByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.bias", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("bias"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsLmHeadBiasByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "lm_head.bias", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("bias"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsBiasTensorWhenOptionEnabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.bias", storage, 0);

    ModelValidationOptions options;
    options.allow_bias = true;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsLoraTensorByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.lora_A.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("lora_A"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsLoraTensorWhenOptionEnabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.lora_A.weight", storage, 0);

    ModelValidationOptions options;
    options.allow_lora_or_adapter = true;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsQuantizedTensorByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.qweight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("qweight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsQuantizedTensorWhenOptionEnabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.q_proj.qweight", storage, 0);

    ModelValidationOptions options;
    options.allow_quantized_tensors = true;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsScalesQuantizedTensor) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.mlp.gate_proj.scales", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("scales"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsNonQuantizedTensorNameContainingScales) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.mlp.gate_proj.scales.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

// --- P0-2: lm_head.weight / tie_word_embeddings linkage (§8.1) ---

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingLmHeadWhenNotTied) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    config.tie_word_embeddings = false;
    RawWeightTable weights = MakeCompleteWeightSet(config, false);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("lm_head.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsMissingLmHeadWhenTied) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    config.tie_word_embeddings = true;
    RawWeightTable weights = MakeCompleteWeightSet(config, false);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsPresentLmHeadWhenTied) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    config.tie_word_embeddings = true;
    RawWeightTable weights = MakeCompleteWeightSet(config, true);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMissingLmHeadWhenRequireLmHeadWhenTiedIsEnabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    config.tie_word_embeddings = true;
    RawWeightTable weights = MakeCompleteWeightSet(config, false);

    ModelValidationOptions options;
    options.require_lm_head_when_tied = true;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("lm_head.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsOutOfRangeLayerIndex) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.1.self_attn.q_proj.weight", storage, 0, {1, 1});

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("layer index 1"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNonNumericLayerIndex) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.bad.self_attn.q_proj.weight", storage, 0, {1, 1});

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("not numeric"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsOutOfRangeNumericLayerIndex) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.9223372036854775808.self_attn.q_proj.weight", storage, 0, {1, 1});

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("out of range"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnsupportedRequiredWeightDType) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& q_proj = weights.at("model.layers.0.self_attn.q_proj.weight");
    q_proj.dtype = DataType::Int(8);
    q_proj.bytes = 1;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Invalid tensor dtype"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsFloat16RequiredWeightDType) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& q_proj = weights.at("model.layers.0.self_attn.q_proj.weight");
    q_proj.dtype = DataType::Float(16);
    q_proj.bytes = 2;
    ModelValidationOptions options;
    options.require_uniform_linear_dtype = false;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsBFloat16RequiredWeightDType) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& q_proj = weights.at("model.layers.0.self_attn.q_proj.weight");
    q_proj.dtype = DataType::BFloat(16);
    q_proj.bytes = 2;
    ModelValidationOptions options;
    options.require_uniform_linear_dtype = false;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsLinearWeightRankOne) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& q_proj = weights.at("model.layers.0.self_attn.q_proj.weight");
    q_proj.shape = {1};

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Invalid tensor rank"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsNormWeightRankTwo) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& norm = weights.at("model.layers.0.input_layernorm.weight");
    norm.shape = {1, 1};

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Invalid tensor rank"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsUnknownTensorByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.extra_metadata.weight", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnknownTensorWhenUnknownTensorsAreDisabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.extra_metadata.weight", storage, 0);
    ModelValidationOptions options;
    options.allow_unknown_tensors = false;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.extra_metadata.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsUnknownTensorWhenStrictTensorNamesIsEnabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.extra_metadata.weight", storage, 0);
    ModelValidationOptions options;
    options.strict_tensor_names = true;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.extra_metadata.weight"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsKnownIgnorableRotaryTensorByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.rotary_emb.inv_freq", storage, 0);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsKnownIgnorableRotaryTensorWhenStrictTensorNamesIsEnabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto storage = std::make_shared<TestStorage>(4);
    AddWeight(&weights, "model.layers.0.self_attn.rotary_emb.inv_freq", storage, 0);
    ModelValidationOptions options;
    options.strict_tensor_names = true;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("rotary_emb.inv_freq"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, RejectsMixedLinearDTypeByDefault) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& q_proj = weights.at("model.layers.0.self_attn.q_proj.weight");
    q_proj.dtype = DataType::Float(16);
    q_proj.bytes = 2;
    auto& k_proj = weights.at("model.layers.0.self_attn.k_proj.weight");
    k_proj.dtype = DataType::BFloat(16);
    k_proj.bytes = 2;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Mixed linear tensor dtype"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AllowsMixedLinearDTypeWhenUniformDTypeIsDisabled) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& q_proj = weights.at("model.layers.0.self_attn.q_proj.weight");
    q_proj.dtype = DataType::Float(16);
    q_proj.bytes = 2;
    auto& k_proj = weights.at("model.layers.0.self_attn.k_proj.weight");
    k_proj.dtype = DataType::BFloat(16);
    k_proj.bytes = 2;
    ModelValidationOptions options;
    options.require_uniform_linear_dtype = false;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights, options);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfModelValidatorTest, RejectsEmbedTokensDTypeMismatchWithLinear) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    auto& embed = weights.at("model.embed_tokens.weight");
    embed.dtype = DataType::Float(16);
    embed.bytes = 2;

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("Mixed linear tensor dtype"), std::string::npos);
}

TEST(ModelLoader_HfModelValidatorTest, AcceptsEmbedTokensDTypeMatchWithLinear) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

}// namespace
}// namespace aethermind
