#include "aethermind/graph/graph_types.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/model/weight/weight_binding_resolver.h"

#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
namespace fs = std::filesystem;

struct AdmissionFixtureCase {
    const char* directory;
    const char* parameter_name;
    DataType linear_dtype;
    DataType norm_dtype;
    const char* hint;
    bool single_norm = false;
    bool tied = false;
};

const std::array<AdmissionFixtureCase, 20> kAdmissionFixtures{{
        {"f32-missing", "F32Missing", DataType::Float32(), DataType::Float32(), ""},
        {"f32-auto", "F32Auto", DataType::Float32(), DataType::Float32(), "auto"},
        {"f32-explicit", "F32Explicit", DataType::Float32(), DataType::Float32(), "float32"},
        {"f16-missing", "F16Missing", DataType::Float(16), DataType::Float(16), ""},
        {"f16-auto", "F16Auto", DataType::Float(16), DataType::Float(16), "auto"},
        {"bf16-missing", "BF16Missing", DataType::BFloat(16), DataType::BFloat(16), ""},
        {"bf16-auto", "BF16Auto", DataType::BFloat(16), DataType::BFloat(16), "auto"},
        {"f16-f32-hint", "F16F32Hint", DataType::Float(16), DataType::Float(16), "float32"},
        {"bf16-f32-hint", "BF16F32Hint", DataType::BFloat(16), DataType::BFloat(16), "float32"},
        {"f32-f16-hint", "F32F16Hint", DataType::Float32(), DataType::Float32(), "float16"},
        {"f32-bf16-hint", "F32BF16Hint", DataType::Float32(), DataType::Float32(), "bfloat16"},
        {"mixed-norm-f16-missing", "MixedNormF16Missing", DataType::Float32(), DataType::Float(16), ""},
        {"mixed-norm-f16-auto", "MixedNormF16Auto", DataType::Float32(), DataType::Float(16), "auto"},
        {"mixed-norm-bf16-missing", "MixedNormBF16Missing", DataType::Float32(), DataType::BFloat(16), ""},
        {"mixed-norm-bf16-auto", "MixedNormBF16Auto", DataType::Float32(), DataType::BFloat(16), "auto"},
        {"single-norm-f16-missing", "SingleNormF16Missing", DataType::Float32(), DataType::Float(16), "", true},
        {"single-norm-f16-auto", "SingleNormF16Auto", DataType::Float32(), DataType::Float(16), "auto", true},
        {"single-norm-bf16-missing", "SingleNormBF16Missing", DataType::Float32(), DataType::BFloat(16), "", true},
        {"single-norm-bf16-auto", "SingleNormBF16Auto", DataType::Float32(), DataType::BFloat(16), "auto", true},
        {"f32-tied", "F32Tied", DataType::Float32(), DataType::Float32(), "float32", false, true},
}};

fs::path AdmissionFixtureDirectory(const char* name) {
    return fs::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / name;
}

void ExpectRawWeight(const RawWeightView& weight, DataType dtype,
                     std::initializer_list<int64_t> shape) {
    EXPECT_TRUE(weight.IsValid());
    EXPECT_TRUE(weight.is_contiguous);
    EXPECT_EQ(weight.dtype, dtype);
    EXPECT_EQ(weight.shape, std::vector<int64_t>(shape));
    const Status valid = ValidateRawWeightView(weight);
    EXPECT_TRUE(valid.ok()) << valid.ToString();
}

class InferenceAdmissionFixture : public ::testing::TestWithParam<AdmissionFixtureCase> {
protected:
    void SetUp() override {
        const fs::path directory = AdmissionFixtureDirectory(GetParam().directory);
        auto loaded = ModelLoader::Load(directory);
        ASSERT_TRUE(loaded.ok())
                << "Generate and check M0 fixtures with tools/generate_fp32_admission_hf_fixtures.py before this test: "
                << directory << ": " << loaded.status().ToString();
        model_ = std::move(*loaded);
    }

    std::unique_ptr<LoadedModel> model_;
};

TEST_P(InferenceAdmissionFixture, PreservesConfigAndDTypeHint) {
    const auto& expected = GetParam();
    const auto& config = model_->GetConfig();
    EXPECT_EQ(config.model_type, "llama");
    EXPECT_EQ(config.architectures, (std::vector<std::string>{"LlamaForCausalLM"}));
    EXPECT_EQ(config.hidden_size, 16);
    EXPECT_EQ(config.intermediate_size, 64);
    EXPECT_EQ(config.num_hidden_layers, 2);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.num_key_value_heads, 4);
    EXPECT_EQ(config.head_dim, 4);
    EXPECT_EQ(config.vocab_size, 32000);
    EXPECT_EQ(config.max_position_embeddings, 2048);
    EXPECT_EQ(config.hidden_act, "silu");
    EXPECT_EQ(config.rope.algorithm, HfRoPEAlgorithm::kStandard);
    EXPECT_DOUBLE_EQ(config.rope.theta, 10000.0);
    EXPECT_EQ(config.tie_word_embeddings, expected.tied);
    EXPECT_EQ(config.weight_dtype_hint_name, expected.hint);

    const std::string hint = expected.hint;
    const DataType expected_hint = hint == "float32"    ? DataType::Float32()
                                   : hint == "float16"  ? DataType::Float(16)
                                   : hint == "bfloat16" ? DataType::BFloat(16)
                                                        : DataType{};
    EXPECT_EQ(config.weight_dtype_hint, expected_hint);
}

TEST_P(InferenceAdmissionFixture, LoadsAllExpectedWeightRolesWithMatchingBytes) {
    const auto& expected = GetParam();
    const auto& config = model_->GetConfig();
    const auto& weights = model_->GetResolvedWeights();
    const int64_t hidden = config.hidden_size;
    const int64_t intermediate = config.intermediate_size;
    const int64_t kv_hidden = config.num_key_value_heads * config.head_dim;

    ExpectRawWeight(weights.embed_tokens, expected.linear_dtype, {config.vocab_size, hidden});
    ExpectRawWeight(weights.final_norm,
                    expected.single_norm ? DataType::Float32() : expected.norm_dtype,
                    {hidden});
    EXPECT_EQ(weights.lm_head.has_value(), !expected.tied);
    if (weights.lm_head.has_value()) {
        ExpectRawWeight(*weights.lm_head, expected.linear_dtype, {config.vocab_size, hidden});
    }

    ASSERT_EQ(weights.layers.size(), static_cast<size_t>(config.num_hidden_layers));
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        SCOPED_TRACE(layer_index);
        const auto& layer = weights.layers[layer_index];
        const DataType input_norm_dtype = expected.single_norm ? DataType::Float32() : expected.norm_dtype;
        const DataType post_norm_dtype = expected.single_norm && layer_index + 1 != weights.layers.size()
                                                 ? DataType::Float32()
                                                 : expected.norm_dtype;
        ExpectRawWeight(layer.norm.input_rmsnorm, input_norm_dtype, {hidden});
        ExpectRawWeight(layer.norm.post_attn_rmsnorm, post_norm_dtype, {hidden});
        ExpectRawWeight(layer.attn.q_proj, expected.linear_dtype, {config.num_attention_heads * config.head_dim, hidden});
        ExpectRawWeight(layer.attn.k_proj, expected.linear_dtype, {kv_hidden, hidden});
        ExpectRawWeight(layer.attn.v_proj, expected.linear_dtype, {kv_hidden, hidden});
        ExpectRawWeight(layer.attn.o_proj, expected.linear_dtype, {hidden, config.num_attention_heads * config.head_dim});
        ExpectRawWeight(layer.mlp.gate_proj, expected.linear_dtype, {intermediate, hidden});
        ExpectRawWeight(layer.mlp.up_proj, expected.linear_dtype, {intermediate, hidden});
        ExpectRawWeight(layer.mlp.down_proj, expected.linear_dtype, {hidden, intermediate});
    }
}

INSTANTIATE_TEST_SUITE_P(
        AdmissionFixtures, InferenceAdmissionFixture, ::testing::ValuesIn(kAdmissionFixtures),
        [](const ::testing::TestParamInfo<AdmissionFixtureCase>& info) {
            return info.param.parameter_name;
        });

TEST(InferenceAdmissionFixtures, TiedLmHeadResolvesToEmbeddingStorage) {
    auto loaded = ModelLoader::Load(AdmissionFixtureDirectory("f32-tied"));
    ASSERT_TRUE(loaded.ok())
            << "Generate M0 fixtures before running this test: " << loaded.status().ToString();
    const auto& weights = (*loaded)->GetResolvedWeights();
    EXPECT_TRUE((*loaded)->GetConfig().tie_word_embeddings);
    EXPECT_FALSE(weights.lm_head.has_value());

    const RawWeightView* head = ResolveWeightBinding(
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kLmHead), weights);
    EXPECT_EQ(head, &weights.embed_tokens);
}

} // namespace
