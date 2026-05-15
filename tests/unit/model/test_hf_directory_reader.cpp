#include "aethermind/model/formats/hf/hf_directory_reader.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <span>
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
                ("aethermind_hf_directory_reader_" + unique_id);
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    AM_NODISCARD const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_{};
};

void WriteTextFile(
        const std::filesystem::path& path,
        std::string_view content) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    stream << content;
}

void WriteBinaryFile(const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    const char zero = 0;
    stream.write(&zero, 1);
}

std::vector<std::byte> ToBytes(std::string_view text) {
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

std::array<std::byte, sizeof(uint64_t)> EncodeLittleEndianU64(uint64_t value) {
    std::array<std::byte, sizeof(uint64_t)> bytes{};
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
    }
    return bytes;
}

std::vector<std::byte> FloatArrayToBytes(std::span<const float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

float ReadFloat(const std::byte* data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(float));
    return value;
}

void WriteRawFile(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
}

void WriteSafetensorsFile(
        const std::filesystem::path& path,
        std::string_view header_json,
        std::span<const std::byte> raw_bytes) {
    const auto prefix = EncodeLittleEndianU64(header_json.size());
    const auto header_bytes = ToBytes(header_json);

    std::vector<std::byte> file_bytes;
    file_bytes.reserve(prefix.size() + header_bytes.size() + raw_bytes.size());
    file_bytes.insert(file_bytes.end(), prefix.begin(), prefix.end());
    file_bytes.insert(file_bytes.end(), header_bytes.begin(), header_bytes.end());
    file_bytes.insert(file_bytes.end(), raw_bytes.begin(), raw_bytes.end());
    WriteRawFile(path, file_bytes);
}

TEST(HfDirectoryReaderTest, InspectSingleFileDirectory) {
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

TEST(HfDirectoryReaderTest, OpensSingleFileReader) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto reader = HfDirectoryReader::Open(temp_dir.path());

    ASSERT_TRUE(reader.ok()) << reader.status().ToString();
    EXPECT_TRUE(reader->GetDirDesc().IsSingleFile());
    EXPECT_EQ(reader->GetDirDesc().config_path, temp_dir.path() / "config.json");
    EXPECT_EQ(reader->GetDirDesc().safetensors_path, temp_dir.path() / "model.safetensors");
}

TEST(HfDirectoryReaderTest, LoadsSingleFileTensorTable) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    WriteSafetensorsFile(
            temp_dir.path() / "model.safetensors",
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    auto reader = HfDirectoryReader::Open(temp_dir.path());
    ASSERT_TRUE(reader.ok()) << reader.status().ToString();

    const auto tensor_table = reader->LoadTensorTable();

    ASSERT_TRUE(tensor_table.ok()) << tensor_table.status().ToString();
    ASSERT_EQ(tensor_table->size(), 1U);
    const auto it = tensor_table->find("weight");
    ASSERT_NE(it, tensor_table->end());
    EXPECT_TRUE(it->second.IsValid());
    EXPECT_EQ(it->second.dtype, DataType::Float32());
    EXPECT_EQ(it->second.shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(it->second.bytes, 8U);
    EXPECT_FLOAT_EQ(ReadFloat(it->second.data), 1.0f);
    EXPECT_FLOAT_EQ(ReadFloat(it->second.data + sizeof(float)), 2.0f);
}

TEST(HfDirectoryReaderTest, RejectsMissingConfigJson) {
    TempDirectory temp_dir;
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kNotFound);
}

TEST(HfDirectoryReaderTest, RejectsMissingSafetensorsFile) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kNotFound);
}

TEST(HfDirectoryReaderTest, RejectsNonDirectoryPath) {
    TempDirectory temp_dir;
    const auto file_path = temp_dir.path() / "config.json";
    WriteTextFile(file_path, "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(file_path);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfDirectoryReaderTest, RejectsConflictingSingleAndShardedLayout) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json", "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kFailedPrecondition);
}

TEST(HfDirectoryReaderTest, ReportsShardedLayoutAsUnimplemented) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json", "{}");

    const auto layout = HfDirectoryReader::InspectDirectory(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kUnimplemented);
}

}// namespace
}// namespace aethermind
