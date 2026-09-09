#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

class TempDirectory {
public:
    TempDirectory() {
        static size_t counter = 0;
        const auto unique_id = std::to_string(
                                       std::chrono::steady_clock::now().time_since_epoch().count()) +
                               "_" + std::to_string(counter++);
        path_ = std::filesystem::temp_directory_path() /
                ("aethermind_hf_config_parser_" + unique_id);
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    TempDirectory(TempDirectory&&) = delete;
    TempDirectory& operator=(TempDirectory&&) = delete;

    AM_NODISCARD const std::filesystem::path& Path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_{};
};

void WriteConfig(const std::filesystem::path& dir, std::string_view content) {
    const auto path = dir / "config.json";
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    stream << content;
}

void WriteMinimalSafetensors(const std::filesystem::path& dir) {
    const auto path = dir / "model.safetensors";
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    const std::string header = "{}";
    constexpr std::array<std::byte, 8> len_bytes = [] {
        std::array<std::byte, 8> bytes{};
        for (size_t i = 0; i < 8; ++i) {
            constexpr uint64_t len = 2;
            bytes[i] = static_cast<std::byte>((len >> (8U * i)) & 0xFFU);
        }
        return bytes;
    }();
    stream.write(reinterpret_cast<const char*>(len_bytes.data()), len_bytes.size());
    stream << header;
}

std::string MakeConfigWithoutRequiredField(std::string_view omitted_field) {
    std::string config = "{\n";
    bool first = true;

    const auto append_field = [&](std::string_view field, std::string_view value) {
        if (field == omitted_field) {
            return;
        }
        if (!first) {
            config += ",\n";
        }
        first = false;
        config += "        \"";
        config += field;
        config += "\": ";
        config += value;
    };

    append_field("architectures", R"(["LlamaForCausalLM"])");
    append_field("model_type", R"("llama")");
    append_field("hidden_size", "4096");
    append_field("intermediate_size", "11008");
    append_field("num_hidden_layers", "32");
    append_field("num_attention_heads", "32");
    append_field("max_position_embeddings", "4096");
    append_field("vocab_size", "32000");
    append_field("rms_norm_eps", "1e-6");

    config += "\n    }";
    return config;
}

StatusOr<HfDirectoryReader> OpenTempDir(const TempDirectory& temp_dir) {
    return HfDirectoryReader::Open(temp_dir.Path());
}

class HfConfigMissingRequiredFieldTest : public ::testing::TestWithParam<const char*> {};

TEST(ModelLoader_HfConfigTest, ParsesMinimalLlamaConfig) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "attention_bias": false,
        "mlp_bias": false,
        "rope_theta": 10000.0,
        "rope_scaling": {"type": "linear", "factor": 2.0},
        "torch_dtype": "bfloat16",
        "unused_field": {"nested": true}
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config->model_type, "llama");
    EXPECT_EQ(config->architectures, (std::vector<std::string>{"LlamaForCausalLM"}));
    EXPECT_EQ(config->hidden_size, 4096);
    EXPECT_EQ(config->intermediate_size, 11008);
    EXPECT_EQ(config->num_hidden_layers, 32);
    EXPECT_EQ(config->num_attention_heads, 32);
    EXPECT_EQ(config->num_key_value_heads, 8);
    EXPECT_EQ(config->vocab_size, 32000);
    EXPECT_EQ(config->max_position_embeddings, 4096);
    EXPECT_EQ(config->head_dim, 128);
    EXPECT_DOUBLE_EQ(config->rms_norm_eps, 1e-6);
    EXPECT_EQ(config->hidden_act, "silu");
    EXPECT_FALSE(config->tie_word_embeddings);
    EXPECT_FALSE(config->attention_bias);
    EXPECT_FALSE(config->mlp_bias);
    EXPECT_DOUBLE_EQ(config->rope.theta, 10000.0);
    ASSERT_TRUE(config->rope.factor.has_value());
    EXPECT_DOUBLE_EQ(*config->rope.factor, 2.0);
    EXPECT_EQ(config->rope.algorithm, HfRoPEAlgorithm::kLinear);
    EXPECT_EQ(config->weight_dtype_hint_name, "bfloat16");
    EXPECT_EQ(config->weight_dtype_hint, DataType::BFloat(16));
}

TEST(ModelLoader_HfConfigTest, DefaultsOptionalFields) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 4096,
        "vocab_size": 32000,
        "rms_norm_eps": 0.000001
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config->num_key_value_heads, config->num_attention_heads);
    EXPECT_EQ(config->head_dim, 0);
    EXPECT_EQ(config->hidden_act, "silu");
    EXPECT_FALSE(config->tie_word_embeddings);
    EXPECT_FALSE(config->attention_bias);
    EXPECT_FALSE(config->mlp_bias);
    EXPECT_DOUBLE_EQ(config->rope.theta, 10000.0);
    EXPECT_FALSE(config->rope.factor.has_value());
    EXPECT_TRUE(config->weight_dtype_hint_name.empty());
    EXPECT_EQ(config->weight_dtype_hint, DataType{});
}

TEST_P(HfConfigMissingRequiredFieldTest, DefersMissingRequiredFieldToValidator) {
    TempDirectory temp_dir;
    const auto* const missing_field = GetParam();
    WriteConfig(temp_dir.Path(), MakeConfigWithoutRequiredField(missing_field));
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();

    const Status status = HfModelValidator::ValidateConfig(*config);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find(missing_field), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
        RequiredFields,
        HfConfigMissingRequiredFieldTest,
        ::testing::Values(
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "max_position_embeddings",
                "vocab_size",
                "rms_norm_eps"),
        [](const ::testing::TestParamInfo<HfConfigMissingRequiredFieldTest::ParamType>& info) {
            return std::string(info.param);
        });

TEST(ModelLoader_HfConfigTest, RejectsMalformedJson) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({"model_type": )");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfConfigTest, RejectsWrongFieldType) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": "4096",
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-6
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfConfigTest, ParsesNullRopeScalingAsAbsent) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_FALSE(config->rope.factor.has_value());
    EXPECT_EQ(config->rope.algorithm, HfRoPEAlgorithm::kStandard);
}

TEST(ModelLoader_HfConfigTest, ParsesDTypeAndRopeParametersAliases) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "dtype": "float16",
        "rope_parameters": {"rope_type": "llama3", "factor": 8.0}
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_TRUE(config->weight_dtype_hint.IsFloat16());
    EXPECT_EQ(config->weight_dtype_hint_name, "float16");
    ASSERT_TRUE(config->rope.factor.has_value());
    EXPECT_DOUBLE_EQ(*config->rope.factor, 8.0);
    EXPECT_EQ(config->rope.algorithm, HfRoPEAlgorithm::kLlama3);
}

TEST(ModelLoader_HfConfigTest, PreservesTopLevelRopeThetaBeforeRopeScaling) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0,
        "rope_scaling": {"type": "llama3", "factor": 8.0}
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_DOUBLE_EQ(config->rope.theta, 500000.0);
    ASSERT_TRUE(config->rope.factor.has_value());
    EXPECT_DOUBLE_EQ(*config->rope.factor, 8.0);
    EXPECT_EQ(config->rope.algorithm, HfRoPEAlgorithm::kLlama3);
}

TEST(ModelLoader_HfConfigTest, ParsesNestedRopeThetaFromRopeScaling) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {"type": "llama3", "factor": 8.0, "rope_theta": 500000.0}
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_DOUBLE_EQ(config->rope.theta, 500000.0);
    ASSERT_TRUE(config->rope.factor.has_value());
    EXPECT_DOUBLE_EQ(*config->rope.factor, 8.0);
    EXPECT_EQ(config->rope.algorithm, HfRoPEAlgorithm::kLlama3);
}

TEST(ModelLoader_HfConfigTest, ParsesExtendedRopeAlgorithmFields) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "original_max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "partial_rotary_factor": 0.5,
        "rope_scaling": {
            "rope_type": "longrope",
            "factor": 8.0,
            "attention_factor": 1.25,
            "short_factor": [1.0, 2.0],
            "long_factor": [4.0, 8.0],
            "truncate": false
        }
    })");
    WriteMinimalSafetensors(temp_dir.Path());
    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();
    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config->rope.algorithm, HfRoPEAlgorithm::kLongRope);
    ASSERT_TRUE(config->rope.partial_rotary_factor.has_value());
    EXPECT_DOUBLE_EQ(*config->rope.partial_rotary_factor, 0.5);
    ASSERT_TRUE(config->rope.original_context_length.has_value());
    EXPECT_EQ(*config->rope.original_context_length, 2048);
    ASSERT_TRUE(config->rope.attention_factor.has_value());
    EXPECT_DOUBLE_EQ(*config->rope.attention_factor, 1.25);
    EXPECT_EQ(config->rope.short_factors, (std::vector<double>{1.0, 2.0}));
    EXPECT_EQ(config->rope.long_factors, (std::vector<double>{4.0, 8.0}));
    ASSERT_TRUE(config->rope.truncate_correction_range.has_value());
    EXPECT_FALSE(*config->rope.truncate_correction_range);
}

TEST(ModelLoader_HfConfigTest, ParsesHfRoPEAlgorithmNames) {
    EXPECT_EQ(ParseHfRoPEAlgorithm(""), HfRoPEAlgorithm::kStandard);
    EXPECT_EQ(ParseHfRoPEAlgorithm("default"), HfRoPEAlgorithm::kStandard);
    EXPECT_EQ(ParseHfRoPEAlgorithm("linear"), HfRoPEAlgorithm::kLinear);
    EXPECT_EQ(ParseHfRoPEAlgorithm("dynamic"), HfRoPEAlgorithm::kDynamicNtk);
    EXPECT_EQ(ParseHfRoPEAlgorithm("dynamic_ntk"), HfRoPEAlgorithm::kDynamicNtk);
    EXPECT_EQ(ParseHfRoPEAlgorithm("yarn"), HfRoPEAlgorithm::kYarn);
    EXPECT_EQ(ParseHfRoPEAlgorithm("llama3"), HfRoPEAlgorithm::kLlama3);
    EXPECT_EQ(ParseHfRoPEAlgorithm("longrope"), HfRoPEAlgorithm::kLongRope);
    EXPECT_EQ(ParseHfRoPEAlgorithm("su"), HfRoPEAlgorithm::kSu);
    EXPECT_EQ(ParseHfRoPEAlgorithm("future_rope"), HfRoPEAlgorithm::kUnknown);
}

TEST(ModelLoader_HfConfigTest, FormatsHfRoPEAlgorithmNames) {
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kStandard), "default");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kLinear), "linear");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kDynamicNtk), "dynamic");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kYarn), "yarn");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kLlama3), "llama3");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kLongRope), "longrope");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kSu), "su");
    EXPECT_EQ(ToString(HfRoPEAlgorithm::kUnknown), "unknown");
}

TEST(ModelLoader_HfConfigTest, DefersUnsupportedDTypeToValidator) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "float8_e4m3fn"
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config->weight_dtype_hint_name, "float8_e4m3fn");
    EXPECT_EQ(config->weight_dtype_hint, DataType{});

    const Status status = HfModelValidator::ValidateConfig(*config);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfConfigTest, SkipsUnknownFloatingJsonValue) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "unused_float": 3.14159
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
}

} // namespace
