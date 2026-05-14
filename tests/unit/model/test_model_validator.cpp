#include "aethermind/model/model_validator.h"

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

void AddTensor(RawTensorMap* tensors, std::string name) {
    tensors->emplace(std::move(name), RawTensorView{});
}

RawTensorMap MakeCompleteTensorSet(const ModelConfig& config) {
    RawTensorMap tensors;
    AddTensor(&tensors, "model.embed_tokens.weight");
    AddTensor(&tensors, "model.norm.weight");

    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        AddTensor(&tensors, prefix + ".self_attn.q_proj.weight");
        AddTensor(&tensors, prefix + ".self_attn.k_proj.weight");
        AddTensor(&tensors, prefix + ".self_attn.v_proj.weight");
        AddTensor(&tensors, prefix + ".self_attn.o_proj.weight");
        AddTensor(&tensors, prefix + ".mlp.gate_proj.weight");
        AddTensor(&tensors, prefix + ".mlp.up_proj.weight");
        AddTensor(&tensors, prefix + ".mlp.down_proj.weight");
        AddTensor(&tensors, prefix + ".input_layernorm.weight");
        AddTensor(&tensors, prefix + ".post_attention_layernorm.weight");
    }

    return tensors;
}

TEST(ModelValidatorTest, AcceptsValidLlamaConfig) {
    const ModelConfig config = MakeValidLlamaConfig();

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelValidatorTest, AcceptsLlamaArchitectureWhenModelTypeIsDifferent) {
    ModelConfig config = MakeValidLlamaConfig();
    config.model_type = "unknown";

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelValidatorTest, RejectsUnsupportedModelFamily) {
    ModelConfig config = MakeValidLlamaConfig();
    config.model_type = "gpt_neox";
    config.architectures = {"GPTNeoXForCausalLM"};

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelValidatorTest, RejectsNonPositiveRequiredDimensions) {
    ModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 0;

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelValidatorTest, RejectsNonPositiveRmsNormEps) {
    ModelConfig config = MakeValidLlamaConfig();
    config.rms_norm_eps = 0.0;

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelValidatorTest, RejectsHiddenSizeNotDivisibleByAttentionHeads) {
    ModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 4097;

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelValidatorTest, RejectsTooManyKeyValueHeads) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_key_value_heads = 33;

    const Status status = ModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelValidatorTest, AcceptsCompleteTensorSet) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    const RawTensorMap tensors = MakeCompleteTensorSet(config);

    const Status status = ModelValidator::ValidateTensorSet(config, tensors);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelValidatorTest, RejectsMissingEmbeddingTensor) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawTensorMap tensors = MakeCompleteTensorSet(config);
    tensors.erase("model.embed_tokens.weight");

    const Status status = ModelValidator::ValidateTensorSet(config, tensors);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.embed_tokens.weight"), std::string::npos);
}

TEST(ModelValidatorTest, RejectsMissingFinalNormTensor) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawTensorMap tensors = MakeCompleteTensorSet(config);
    tensors.erase("model.norm.weight");

    const Status status = ModelValidator::ValidateTensorSet(config, tensors);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.norm.weight"), std::string::npos);
}

TEST(ModelValidatorTest, RejectsMissingLayerAttentionTensor) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    RawTensorMap tensors = MakeCompleteTensorSet(config);
    tensors.erase("model.layers.1.self_attn.q_proj.weight");

    const Status status = ModelValidator::ValidateTensorSet(config, tensors);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.1.self_attn.q_proj.weight"), std::string::npos);
}

TEST(ModelValidatorTest, RejectsMissingLayerMlpTensor) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawTensorMap tensors = MakeCompleteTensorSet(config);
    tensors.erase("model.layers.0.mlp.down_proj.weight");

    const Status status = ModelValidator::ValidateTensorSet(config, tensors);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.mlp.down_proj.weight"), std::string::npos);
}

TEST(ModelValidatorTest, RejectsMissingLayerNormTensor) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawTensorMap tensors = MakeCompleteTensorSet(config);
    tensors.erase("model.layers.0.post_attention_layernorm.weight");

    const Status status = ModelValidator::ValidateTensorSet(config, tensors);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.post_attention_layernorm.weight"), std::string::npos);
}

TEST(ModelValidatorTest, RejectsTensorSetWithInvalidLayerCount) {
    ModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 0;

    const Status status = ModelValidator::ValidateTensorSet(config, RawTensorMap{});

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
