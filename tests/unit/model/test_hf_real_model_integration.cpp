#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/raw_weight.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace aethermind {
namespace {
namespace fs = std::filesystem;

fs::path TestModelDir() {
    return fs::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM";
}

TEST(ModelLoader_HfRealModelIntegrationTest, InspectFindsSingleSafetensorsLayout) {
    const auto layout = HfDirectoryReader::InspectDirectory(TestModelDir());

    ASSERT_TRUE(layout.ok()) << layout.status().ToString();
    EXPECT_EQ(layout->layout, HfDirectoryLayout::kSingleSafetensors);
    EXPECT_EQ(layout->config_path, TestModelDir() / "config.json");
    EXPECT_EQ(layout->safetensors_path, TestModelDir() / "model.safetensors");
    EXPECT_TRUE(layout->safetensors_index_path.empty());
}

TEST(ModelLoader_HfRealModelIntegrationTest, ParsesRealConfigCorrectly) {
    auto reader = HfDirectoryReader::Open(TestModelDir());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    auto config = reader->ParseConfig();
    ASSERT_TRUE(config.ok()) << config.status().ToString();

    EXPECT_EQ(config->model_type, "llama");
    ASSERT_EQ(config->architectures.size(), 1U);
    EXPECT_EQ(config->architectures[0], "LlamaForCausalLM");

    EXPECT_EQ(config->hidden_size, 16);
    EXPECT_EQ(config->intermediate_size, 64);
    EXPECT_EQ(config->num_hidden_layers, 2);
    EXPECT_EQ(config->num_attention_heads, 4);
    EXPECT_EQ(config->num_key_value_heads, 4);
    EXPECT_EQ(config->head_dim, 4);
    EXPECT_EQ(config->vocab_size, 32000);
    EXPECT_EQ(config->max_position_embeddings, 2048);

    EXPECT_EQ(config->hidden_act, "silu");
    EXPECT_DOUBLE_EQ(config->rms_norm_eps, 1e-6);
    EXPECT_FALSE(config->tie_word_embeddings);
    EXPECT_FALSE(config->attention_bias);
    EXPECT_FALSE(config->mlp_bias);

    EXPECT_EQ(config->weight_dtype_hint_name, "float32");
    EXPECT_TRUE(config->weight_dtype_hint.IsFloat32());

    EXPECT_DOUBLE_EQ(config->rope.theta, 10000.0);
    EXPECT_FALSE(config->rope.scaling_factor.has_value());
    EXPECT_TRUE(config->rope.scaling_type.empty());
}

TEST(ModelLoader_HfRealModelIntegrationTest, LoadsRealWeightTable) {
    auto reader = HfDirectoryReader::Open(TestModelDir());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    auto raw_weights = reader->LoadRawWeightTable();
    ASSERT_TRUE(raw_weights.ok()) << raw_weights.status().ToString();

    EXPECT_GE(raw_weights->size(), 1U);

    ASSERT_TRUE(raw_weights->contains("model.embed_tokens.weight"));
    ASSERT_TRUE(raw_weights->contains("model.norm.weight"));
    ASSERT_TRUE(raw_weights->contains("lm_head.weight"));

    ASSERT_TRUE(raw_weights->contains("model.layers.0.self_attn.q_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.self_attn.k_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.self_attn.v_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.self_attn.o_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.mlp.gate_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.mlp.up_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.mlp.down_proj.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.input_layernorm.weight"));
    ASSERT_TRUE(raw_weights->contains("model.layers.0.post_attention_layernorm.weight"));

    ASSERT_TRUE(raw_weights->contains("model.layers.1.self_attn.q_proj.weight"));

    const auto& embed = raw_weights->at("model.embed_tokens.weight");
    EXPECT_TRUE(embed.IsValid());
    EXPECT_EQ(embed.dtype, DataType::Float32());
    EXPECT_EQ(embed.shape, (std::vector<int64_t>{32000, 16}));

    const auto& q_proj = raw_weights->at("model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(q_proj.shape, (std::vector<int64_t>{16, 16}));

    const auto& gate = raw_weights->at("model.layers.0.mlp.gate_proj.weight");
    EXPECT_EQ(gate.shape, (std::vector<int64_t>{64, 16}));

    const auto& norm = raw_weights->at("model.layers.0.input_layernorm.weight");
    EXPECT_EQ(norm.shape, (std::vector<int64_t>{16}));
}

TEST(ModelLoader_HfRealModelIntegrationTest, ValidatesConfigAndWeightSet) {
    auto reader = HfDirectoryReader::Open(TestModelDir());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    auto config = reader->ParseConfig();
    ASSERT_TRUE(config.ok()) << config.status().ToString();

    auto raw_weights = reader->LoadRawWeightTable();
    ASSERT_TRUE(raw_weights.ok()) << raw_weights.status().ToString();

    auto config_status = HfModelValidator::ValidateConfig(*config);
    EXPECT_TRUE(config_status.ok()) << config_status.ToString();

    auto weight_status = HfModelValidator::ValidateWeightSet(*config, *raw_weights);
    EXPECT_TRUE(weight_status.ok()) << weight_status.ToString();
}

}// namespace
}// namespace aethermind
