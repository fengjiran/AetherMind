#include "aethermind/model/formats/hf/hf_config_parser.h"

#include <chrono>
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

std::filesystem::path WriteConfig(TempDirectory& temp_dir, std::string_view content) {
    const auto path = temp_dir.Path() / "config.json";
    std::ofstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open()) << path.string();
    stream << content;
    return path;
}

TEST(HfConfigParserTest, ParsesMinimalLlamaConfig) {
    TempDirectory temp_dir;
    const auto path = WriteConfig(temp_dir, R"({
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

    const auto config = HfConfigParser::ParseConfigFile(path);

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

TEST(HfConfigParserTest, DefaultsOptionalFields) {
    TempDirectory temp_dir;
    const auto path = WriteConfig(temp_dir, R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 0.000001
    })");

    const auto config = HfConfigParser::ParseConfigFile(path);

    ASSERT_TRUE(config.ok()) << config.status().ToString();
    EXPECT_EQ(config->num_key_value_heads, config->num_attention_heads);
    EXPECT_FALSE(config->tie_word_embeddings);
}

TEST(HfConfigParserTest, RejectsMissingRequiredField) {
    TempDirectory temp_dir;
    const auto path = WriteConfig(temp_dir, R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-6
    })");

    const auto config = HfConfigParser::ParseConfigFile(path);

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfConfigParserTest, RejectsMalformedJson) {
    TempDirectory temp_dir;
    const auto path = WriteConfig(temp_dir, R"({"model_type": )");

    const auto config = HfConfigParser::ParseConfigFile(path);

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfConfigParserTest, RejectsWrongFieldType) {
    TempDirectory temp_dir;
    const auto path = WriteConfig(temp_dir, R"({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": "4096",
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-6
    })");

    const auto config = HfConfigParser::ParseConfigFile(path);

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfConfigParserTest, RejectsMissingFile) {
    TempDirectory temp_dir;

    const auto config = HfConfigParser::ParseConfigFile(temp_dir.Path() / "config.json");

    ASSERT_FALSE(config.ok());
    EXPECT_EQ(config.status().code(), StatusCode::kNotFound);
}

}// namespace
}// namespace aethermind
