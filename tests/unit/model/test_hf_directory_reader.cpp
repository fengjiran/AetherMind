#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_safetensors_index.h"
#include "test_utils.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>

namespace aethermind {
namespace {

TEST(ModelLoader_HfDirectoryReaderTest, InspectSingleFileDirectory) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_TRUE(layout.ok()) << layout.status().ToString();
    EXPECT_TRUE(layout->IsSingleFile());
    EXPECT_EQ(layout->layout, HfDirectoryLayout::kSingleSafetensors);
    EXPECT_EQ(layout->config_path, temp_dir.path() / "config.json");
    EXPECT_EQ(layout->safetensors_path, temp_dir.path() / "model.safetensors");
    EXPECT_TRUE(layout->safetensors_index_path.empty());
}

TEST(ModelLoader_HfDirectoryReaderTest, OpensSingleFileReader) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto reader = HfDirectoryReader::Open(temp_dir.path());

    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    EXPECT_TRUE(reader->GetDirDesc().IsSingleFile());
    EXPECT_EQ(reader->GetDirDesc().config_path, temp_dir.path() / "config.json");
    EXPECT_EQ(reader->GetDirDesc().safetensors_path, temp_dir.path() / "model.safetensors");
}

TEST(ModelLoader_HfDirectoryReaderTest, LoadsSingleFileRawWeightTable) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    WriteSafetensorsFile(temp_dir.path() / "model.safetensors",
                         R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                         raw_bytes);

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_TRUE(raw_weights.ok()) << raw_weights.status().ToString();
    ASSERT_EQ(raw_weights->size(), 1U);
    const auto it = raw_weights->find("weight");
    ASSERT_NE(it, raw_weights->end());
    EXPECT_TRUE(it->second.IsValid());
    EXPECT_EQ(it->second.dtype, DataType::Float32());
    EXPECT_EQ(it->second.shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(it->second.bytes, 8U);
    EXPECT_FLOAT_EQ(ReadFloat(it->second.data), 1.0f);
    EXPECT_FLOAT_EQ(ReadFloat(it->second.data + sizeof(float)), 2.0f);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsMissingConfigJson) {
    TempDirectory temp_dir;
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kNotFound);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsConfigJsonThatIsNotARegularFile) {
    TempDirectory temp_dir;
    WriteBinaryFile(temp_dir.path() / "model.safetensors");
    std::filesystem::create_directory(temp_dir.path() / "config.json");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsMissingSafetensorsFile) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kNotFound);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsNonDirectoryPath) {
    TempDirectory temp_dir;
    const auto file_path = temp_dir.path() / "config.json";
    WriteTextFile(file_path, "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(file_path);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsConflictingSingleAndShardedLayout) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json", "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ModelLoader_HfDirectoryReaderTest, InspectShardedDirectory) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json", "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_TRUE(layout.ok()) << layout.status().ToString();
    EXPECT_TRUE(layout->IsSharded());
    EXPECT_EQ(layout->layout, HfDirectoryLayout::kShardedSafetensors);
    EXPECT_EQ(layout->config_path, temp_dir.path() / "config.json");
    EXPECT_TRUE(layout->safetensors_path.empty());
    EXPECT_EQ(layout->safetensors_index_path, temp_dir.path() / "model.safetensors.index.json");
}

TEST(ModelLoader_HfDirectoryReaderTest, ParsesSafetensorsIndexMetadataTotalSize) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"metadata":{"total_size":"12"},"weight_map":{"tensor_a":"model-00001-of-00001.safetensors"}})");

    const auto index = HfSafetensorsIndex::Load(temp_dir.path() / "model.safetensors.index.json");

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    ASSERT_TRUE(index->TotalSize().has_value());
    EXPECT_EQ(*index->TotalSize(), 12U);
    ASSERT_EQ(index->WeightMap().size(), 1U);
    EXPECT_EQ(index->WeightMap().at("tensor_a"), "model-00001-of-00001.safetensors");
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsShardedIndexPathThatIsNotARegularFile) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    std::filesystem::create_directory(temp_dir.path() / "model.safetensors.index.json");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfDirectoryReaderTest, LoadsShardedRawWeightTable) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"metadata":{"total_size":12},"weight_map":{"tensor_a":"model-00001-of-00002.safetensors","tensor_b":"model-00002-of-00002.safetensors"}})");
    WriteSafetensorsFileWithFloats(temp_dir.path() / "model-00001-of-00002.safetensors",
                                   R"({"tensor_a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                                   std::array<float, 1>{1.0f});
    WriteSafetensorsFileWithFloats(temp_dir.path() / "model-00002-of-00002.safetensors",
                                   R"({"tensor_b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                                   std::array<float, 2>{2.0f, 3.0f});

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_TRUE(raw_weights.ok()) << raw_weights.status().ToString();
    ASSERT_EQ(raw_weights->size(), 2U);

    const auto tensor_a = raw_weights->find("tensor_a");
    ASSERT_NE(tensor_a, raw_weights->end());
    EXPECT_TRUE(tensor_a->second.IsValid());
    EXPECT_EQ(tensor_a->second.dtype, DataType::Float32());
    EXPECT_EQ(tensor_a->second.shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(tensor_a->second.bytes, 4U);
    EXPECT_FLOAT_EQ(ReadFloat(tensor_a->second.data), 1.0f);

    const auto tensor_b = raw_weights->find("tensor_b");
    ASSERT_NE(tensor_b, raw_weights->end());
    EXPECT_TRUE(tensor_b->second.IsValid());
    EXPECT_EQ(tensor_b->second.dtype, DataType::Float32());
    EXPECT_EQ(tensor_b->second.shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(tensor_b->second.bytes, 8U);
    EXPECT_FLOAT_EQ(ReadFloat(tensor_b->second.data), 2.0f);
    EXPECT_FLOAT_EQ(ReadFloat(tensor_b->second.data + sizeof(float)), 3.0f);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsMissingShardFile) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"weight_map":{"tensor_a":"model-00001-of-00001.safetensors"}})");

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_FALSE(raw_weights.ok());
    EXPECT_EQ(raw_weights.status().code(), StatusCode::kNotFound);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsShardTensorNotListedInIndex) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"weight_map":{"tensor_a":"model-00001-of-00001.safetensors"}})");
    WriteSafetensorsFileWithFloats(temp_dir.path() / "model-00001-of-00001.safetensors",
                                   R"({"tensor_a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"extra":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
                                   std::array<float, 2>{1.0f, 2.0f});

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_FALSE(raw_weights.ok());
    EXPECT_EQ(raw_weights.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsShardAssignmentMismatch) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"weight_map":{"tensor_a":"model-00001-of-00002.safetensors","tensor_b":"model-00002-of-00002.safetensors"}})");
    WriteSafetensorsFileWithFloats(temp_dir.path() / "model-00001-of-00002.safetensors",
                                   R"({"tensor_a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                                   std::array<float, 1>{1.0f});
    WriteSafetensorsFileWithFloats(temp_dir.path() / "model-00002-of-00002.safetensors",
                                   R"({"tensor_a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                                   std::array<float, 1>{2.0f});

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_FALSE(raw_weights.ok());
    EXPECT_EQ(raw_weights.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsMissingTensorInShard) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"weight_map":{"tensor_a":"model-00001-of-00002.safetensors","tensor_b":"model-00002-of-00002.safetensors"}})");
    WriteSafetensorsFileWithFloats(temp_dir.path() / "model-00001-of-00002.safetensors",
                                   R"({"tensor_a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                                   std::array<float, 1>{1.0f});
    WriteSafetensorsFile(temp_dir.path() / "model-00002-of-00002.safetensors", R"({})", std::span<const std::byte>{});

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_FALSE(raw_weights.ok());
    EXPECT_EQ(raw_weights.status().code(), StatusCode::kNotFound);
}

TEST(ModelLoader_HfDirectoryReaderTest, RejectsShardSymlinkEscape) {
    TempDirectory temp_dir;
    TempDirectory outside_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json",
                  R"({"weight_map":{"tensor_a":"model-00001-of-00001.safetensors"}})");
    WriteSafetensorsFileWithFloats(outside_dir.path() / "outside.safetensors",
                                   R"({"tensor_a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                                   std::array<float, 1>{1.0f});

    std::error_code error;
    std::filesystem::create_symlink(outside_dir.path() / "outside.safetensors",
                                    temp_dir.path() / "model-00001-of-00001.safetensors",
                                    error);
    if (error) {
        GTEST_SKIP() << "Unable to create symlink: " << error.message();
    }

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto raw_weights = reader->LoadRawWeightTable();

    ASSERT_FALSE(raw_weights.ok());
    EXPECT_EQ(raw_weights.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
