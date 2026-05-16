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
            .rms_norm_eps = 1e-6,
            .tie_word_embeddings = false,
    };
}

void AddWeight(RawWeightTable* weights,
               std::string name,
               const std::shared_ptr<TestStorage>& storage,
               size_t offset) {
    weights->emplace(std::move(name),
                     RawWeightView{
                             .data = storage->data.data() + offset,
                             .bytes = 4,
                             .dtype = DataType::Float32(),
                             .shape = {1},
                             .storage = storage,
                     });
}

RawWeightTable MakeCompleteWeightSet(const HfModelConfig& config) {
    auto storage = std::make_shared<TestStorage>((config.num_hidden_layers * 9 + 2) * 4);

    RawWeightTable weights;
    size_t offset = 0;
    AddWeight(&weights, "model.embed_tokens.weight", storage, offset);
    offset += 4;
    AddWeight(&weights, "model.norm.weight", storage, offset);
    offset += 4;

    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        AddWeight(&weights, prefix + ".self_attn.q_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".self_attn.k_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".self_attn.v_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".self_attn.o_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".mlp.gate_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".mlp.up_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".mlp.down_proj.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".input_layernorm.weight", storage, offset);
        offset += 4;
        AddWeight(&weights, prefix + ".post_attention_layernorm.weight", storage, offset);
        offset += 4;
    }

    return weights;
}

TEST(HfModelValidatorTest, AcceptsValidLlamaConfig) {
    const HfModelConfig config = MakeValidLlamaConfig();

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(HfModelValidatorTest, AcceptsLlamaArchitectureWhenModelTypeIsDifferent) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.model_type = "unknown";

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(HfModelValidatorTest, RejectsUnsupportedModelFamily) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.model_type = "gpt_neox";
    config.architectures = {"GPTNeoXForCausalLM"};

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsNonPositiveRequiredDimensions) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsNonPositiveRmsNormEps) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.rms_norm_eps = 0.0;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsHiddenSizeNotDivisibleByAttentionHeads) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.hidden_size = 4097;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsTooManyKeyValueHeads) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_key_value_heads = 33;

    const Status status = HfModelValidator::ValidateConfig(config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, AcceptsCompleteWeightSet) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    const RawWeightTable weights = MakeCompleteWeightSet(config);

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(HfModelValidatorTest, RejectsMissingEmbeddingWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.embed_tokens.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.embed_tokens.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingFinalNormWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.norm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.norm.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingLayerAttentionWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 2;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.1.self_attn.q_proj.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.1.self_attn.q_proj.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingLayerMlpWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.0.mlp.down_proj.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.mlp.down_proj.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsMissingLayerNormWeight) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    RawWeightTable weights = MakeCompleteWeightSet(config);
    weights.erase("model.layers.0.post_attention_layernorm.weight");

    const Status status = HfModelValidator::ValidateWeightSet(config, weights);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("model.layers.0.post_attention_layernorm.weight"), std::string::npos);
}

TEST(HfModelValidatorTest, RejectsWeightSetWithInvalidLayerCount) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 0;

    const Status status = HfModelValidator::ValidateWeightSet(config, RawWeightTable{});

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(HfModelValidatorTest, RejectsWeightWithMismatchedByteSize) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    auto storage = std::make_shared<TestStorage>(4);

    RawWeightTable weights;
    weights.emplace("model.embed_tokens.weight",
                    RawWeightView{
                            .data = storage->data.data(),
                            .bytes = 8,  // shape [1] × Float32 = 4, not 8
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

TEST(HfModelValidatorTest, RejectsWeightWithNonPositiveShapeDimension) {
    HfModelConfig config = MakeValidLlamaConfig();
    config.num_hidden_layers = 1;
    auto storage = std::make_shared<TestStorage>(4);

    RawWeightTable weights;
    weights.emplace("model.embed_tokens.weight",
                    RawWeightView{
                            .data = storage->data.data(),
                            .bytes = 0,
                            .dtype = DataType::Float32(),
                            .shape = {0},  // non-positive
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

TEST(HfModelValidatorTest, RejectsWeightWithNullStorage) {
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

}// namespace
}// namespace aethermind
