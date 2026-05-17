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

namespace aethermind {
namespace {

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
    const std::array<std::byte, 8> len_bytes = []() {
        const uint64_t len = 2;
        std::array<std::byte, 8> bytes{};
        for (size_t i = 0; i < 8; ++i) {
            bytes[i] = static_cast<std::byte>((len >> (8U * i)) & 0xFFU);
        }
        return bytes;
    }();
    stream.write(reinterpret_cast<const char*>(len_bytes.data()), static_cast<std::streamsize>(len_bytes.size()));
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
    append_field("vocab_size", "32000");
    append_field("rms_norm_eps", "1e-6");

    config += "\n    }";
    return config;
}

StatusOr<HfDirectoryReader> OpenTempDir(TempDirectory& temp_dir) {
    return HfDirectoryReader::Open(temp_dir.Path());
}

class HfConfigMissingRequiredFieldTest : public ::testing::TestWithParam<const char*> {};

TEST(HfConfigTest, ParsesMinimalLlamaConfig) {
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
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
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
    EXPECT_DOUBLE_EQ(config->rms_norm_eps, 1e-6);
    EXPECT_FALSE(config->tie_word_embeddings);
}

TEST(HfConfigTest, DefaultsOptionalFields) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 0.000001
    })");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config->num_key_value_heads, config->num_attention_heads);
    EXPECT_FALSE(config->tie_word_embeddings);
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
                "architectures",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "vocab_size",
                "rms_norm_eps"),
        [](const ::testing::TestParamInfo<HfConfigMissingRequiredFieldTest::ParamType>& info) {
            return std::string(info.param);
        });

TEST(HfConfigTest, RejectsMalformedJson) {
    TempDirectory temp_dir;
    WriteConfig(temp_dir.Path(), R"({"model_type": )");
    WriteMinimalSafetensors(temp_dir.Path());

    auto reader = OpenTempDir(temp_dir);
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    const auto config = reader->ParseConfig();

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfConfigTest, RejectsWrongFieldType) {
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

}// namespace
}// namespace aethermind
