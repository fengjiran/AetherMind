#include "aethermind/model/formats/hf/hf_model_validator.h"

#include <gtest/gtest.h>

#include <string>

namespace aethermind {
namespace {

ModelConfig MakeValidLlamaConfig() {
    return ModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 4096,
            .intermediate_size = 11008,
            .num_hidden_layers = 32,
            .num_attention_heads = 32,
            .num_key_value_heads = 8,
            .vocab_size = 32000,
            .rms_norm_eps = 1e-6,
            .tie_word_embeddings = false,
    };
}

void AddWeight(RawWeightTable* weights, std::string name) {
    weights->emplace(std::move(name), RawWeightView{});
}

RawWeightTable MakeCompleteWeightSet(const ModelConfig& config) {
    RawWeightTable weights;
    AddWeight(&weights, "model.embed_tokens.weight");
    AddWeight(&weights, "model.norm.weight");

    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        AddWeight(&weights, prefix + ".self_attn.q_proj.weight");
        AddWeight(&weights, prefix + ".self_attn.k_proj.weight");
        AddWeight(&weights, prefix + ".self_attn.v_proj.weight");
        AddWeight(&weights, prefix + ".self_attn.o_proj.weight");
        AddWeight(&weights, prefix + ".mlp.gate_proj.weight");
        AddWeight(&weights, prefix + ".mlp.up_proj.weight");
        AddWeight(&weights, prefix + ".mlp.down_proj.weight");
        AddWeight(&weights, prefix + ".input_layernorm.weight");
        AddWeight(&weights, prefix + ".post_attention_layernorm.weight");
    }

    return weights;
}

TEST(HfModelValidatorTest, AcceptsValidLlamaConfig) {
    const ModelConfig config = MakeValidLlamaConfig();

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(HfModelValidatorTest, AcceptsLlamaArchitectureWhenModelTypeIsDifferent) {
    ModelConfig config = MakeValidLlamaConfig();
    config.model_type = "unknown";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(HfModelValidatorTest, RejectsUnsupportedModelFamily) {
    ModelConfig config = MakeValidLlamaConfig();
    config.model_type = "gpt_neox";
    config.architectures = {"GPTNeoXForCausalLM"};

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsNonPositiveRequiredDimensions) {
    ModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsNonPositiveRmsNormEps) {
    ModelConfig config = MakeValidLlamaConfig();
    config.rms_norm_eps = 0.0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsHiddenSizeNotDivisibleByAttentionHeads) {
    ModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 4097;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsTooManyKeyValueHeads) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_key_value_heads = 33;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, AcceptsCompleteWeightSet) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    const RawWeightTable weights = MakeCompleteWeightSet(config);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(HfModelValidatorTest, RejectsMissingEmbeddingWeight) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.embed_tokens.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.embed_tokens.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingFinalNormWeight) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.norm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.norm.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingLayerAttentionWeight) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.1.self_attn.q_proj.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.1.self_attn.q_proj.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingLayerMlpWeight) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.0.mlp.down_proj.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.mlp.down_proj.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingLayerNormWeight) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.0.post_attention_layernorm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.post_attention_layernorm.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsWeightSetWithInvalidLayerCount) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 0;

    const Status status = HfModelValidator::ValidateWeightSet(config, RawWeightTable{});

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
