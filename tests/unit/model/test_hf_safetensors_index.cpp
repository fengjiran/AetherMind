#include "aethermind/model/formats/hf/hf_safetensors_index.h"
#include "test_utils.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

void WriteIndex(const std::filesystem::path& dir, std::string_view json) {
    WriteTextFile(dir / "model.safetensors.index.json", json);
}

TEST(ModelLoader_HfSafetensorsIndexTest, LoadsValidIndexWithTotalSize) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "metadata": {"total_size": "123456"},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
        }
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_EQ(index->WeightMap().size(), 2U);
    EXPECT_EQ(index->WeightMap().at("model.embed_tokens.weight"),
              "model-00001-of-00002.safetensors");
    EXPECT_EQ(index->TotalSize(), std::optional<uint64_t>(123456));
    EXPECT_EQ(index->UniqueShardFilenames(),
              (std::vector<std::string>{
                      "model-00001-of-00002.safetensors",
                      "model-00002-of-00002.safetensors"}));
}

TEST(ModelLoader_HfSafetensorsIndexTest, LoadsIndexWithoutMetadata) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(),
               R"({"weight_map": {"tensor_a": "model-00001-of-00001.safetensors"}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_FALSE(index->TotalSize().has_value());
}

TEST(ModelLoader_HfSafetensorsIndexTest, LoadsIndexWithNumericTotalSize) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "metadata": {"total_size": 42},
        "weight_map": {"tensor_a": "model-00001-of-00001.safetensors"}
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_EQ(index->TotalSize(), std::optional<uint64_t>(42));
}

TEST(ModelLoader_HfSafetensorsIndexTest, SkipsUnknownTopLevelFields) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "version": 2,
        "unknown_nested": {"a": [1, "x", null]},
        "weight_map": {"tensor_a": "model-00001-of-00001.safetensors"}
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_EQ(index->WeightMap().size(), 1U);
}

TEST(ModelLoader_HfSafetensorsIndexTest, UniqueShardFilenamesDedupesAndSorts) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "weight_map": {
            "tensor_a": "z.safetensors",
            "tensor_b": "a.safetensors",
            "tensor_c": "z.safetensors"
        }
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_EQ(index->UniqueShardFilenames(),
              (std::vector<std::string>{"a.safetensors", "z.safetensors"}));
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsMissingIndexFile) {
    TempDirectory temp_dir;

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "missing.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kNotFound);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsMissingWeightMap) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({"metadata": {"total_size": "1"}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("missing required weight_map"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsEmptyWeightMap) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({"weight_map": {}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("must not be empty"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsDuplicateWeightMapField) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "weight_map": {"tensor_a": "a.safetensors"},
        "weight_map": {"tensor_b": "b.safetensors"}
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("duplicate weight_map field"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsDuplicateMetadataField) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "metadata": {"total_size": "1"},
        "metadata": {"total_size": "2"},
        "weight_map": {"tensor_a": "a.safetensors"}
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("duplicate metadata field"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsDuplicateTotalSize) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "metadata": {"total_size": "1", "total_size": "2"},
        "weight_map": {"tensor_a": "a.safetensors"}
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("duplicate total_size field"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsInvalidTotalSizeString) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "metadata": {"total_size": "not-a-number"},
        "weight_map": {"tensor_a": "a.safetensors"}
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("total_size string is not a valid"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsTrailingJsonContent) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(),
               R"({"weight_map": {"tensor_a": "a.safetensors"}} garbage)");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("trailing JSON content"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsEmptyShardFilename) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({"weight_map": {"tensor_a": ""}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("must not be empty"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsUriSchemeShardFilename) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(),
               R"({"weight_map": {"tensor_a": "https://evil.example/model.safetensors"}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("URI schemes"), std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsAbsoluteShardFilename) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(),
               R"({"weight_map": {"tensor_a": "/etc/passwd"}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("must be relative"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsDotDotShardFilename) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(),
               R"({"weight_map": {"tensor_a": "../outside/model.safetensors"}})");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("must not contain '..'"),
              std::string::npos);
}

TEST(ModelLoader_HfSafetensorsIndexTest, RejectsDuplicateTensorName) {
    TempDirectory temp_dir;
    WriteIndex(temp_dir.path(), R"({
        "weight_map": {
            "tensor_a": "a.safetensors",
            "tensor_a": "b.safetensors"
        }
    })");

    const auto index = HfSafetensorsIndex::Load(
            temp_dir.path() / "model.safetensors.index.json");

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(index.status().message().find("duplicate tensor name"),
              std::string::npos);
}

} // namespace