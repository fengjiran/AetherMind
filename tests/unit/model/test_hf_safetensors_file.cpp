#include "aethermind/model/formats/hf/hf_safetensors_file.h"

#include <array>
#include <chrono>
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
                ("aethermind_hf_safetensors_" + unique_id);
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

std::array<std::byte, sizeof(uint64_t)> EncodeLittleEndianU64(uint64_t value) {
    std::array<std::byte, sizeof(uint64_t)> bytes{};
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
    }
    return bytes;
}

std::vector<std::byte> ToBytes(std::string_view text) {
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
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

void WriteRawFile(
        const std::filesystem::path& path,
        std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
}

std::filesystem::path WriteSafetensorsFile(
        const std::filesystem::path& directory,
        std::string_view header_json,
        std::span<const std::byte> raw_bytes) {
    const auto path = directory / "model.safetensors";
    const auto prefix = EncodeLittleEndianU64(header_json.size());
    const auto header_bytes = ToBytes(header_json);

    std::vector<std::byte> file_bytes;
    file_bytes.reserve(prefix.size() + header_bytes.size() + raw_bytes.size());
    file_bytes.insert(file_bytes.end(), prefix.begin(), prefix.end());
    file_bytes.insert(file_bytes.end(), header_bytes.begin(), header_bytes.end());
    file_bytes.insert(file_bytes.end(), raw_bytes.begin(), raw_bytes.end());
    WriteRawFile(path, file_bytes);
    return path;
}

TEST(ModelLoader_HfSafetensorsFileTest, LoadsSingleTensorFile) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_TRUE(file.ok()) << file.status().ToString();
    ASSERT_EQ(file->Entries().size(), 1U);
    const auto* entry = file->Find("weight");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->dtype, DataType::Float32());
    EXPECT_EQ(entry->shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(entry->data_offset_begin, 0U);
    EXPECT_EQ(entry->data_offset_end, 16U);
    EXPECT_TRUE(entry->view.IsValid());
    EXPECT_FLOAT_EQ(ReadFloat(entry->view.data), 1.0f);
    EXPECT_FLOAT_EQ(ReadFloat(entry->view.data + 3 * sizeof(float)), 4.0f);
}

TEST(ModelLoader_HfSafetensorsFileTest, LoadsMultipleTensorEntries) {
    TempDirectory temp_dir;
    const auto w1_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto w2_bytes = FloatArrayToBytes(std::array<float, 2>{3.0f, 4.0f});
    std::vector<std::byte> raw_bytes;
    raw_bytes.insert(raw_bytes.end(), w1_bytes.begin(), w1_bytes.end());
    raw_bytes.insert(raw_bytes.end(), w2_bytes.begin(), w2_bytes.end());

    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"w1":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"w2":{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_TRUE(file.ok()) << file.status().ToString();
    EXPECT_EQ(file->Entries().size(), 2U);
    ASSERT_NE(file->Find("w1"), nullptr);
    ASSERT_NE(file->Find("w2"), nullptr);
}

TEST(ModelLoader_HfSafetensorsFileTest, RejectsInvalidHeaderLength) {
    TempDirectory temp_dir;
    std::vector<std::byte> bytes;
    const auto prefix = EncodeLittleEndianU64(1024);
    bytes.insert(bytes.end(), prefix.begin(), prefix.end());
    bytes.push_back(static_cast<std::byte>('{'));
    const auto path = temp_dir.Path() / "model.safetensors";
    WriteRawFile(path, bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_FALSE(file.ok());
    EXPECT_EQ(file.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfSafetensorsFileTest, RejectsMalformedHeaderJson) {
    TempDirectory temp_dir;
    const auto header = R"({"weight":)";
    const auto path = WriteSafetensorsFile(temp_dir.Path(), header, {});

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_FALSE(file.ok());
    EXPECT_EQ(file.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfSafetensorsFileTest, RejectsUnknownDType) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F128","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_FALSE(file.ok());
    EXPECT_EQ(file.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfSafetensorsFileTest, RejectsOutOfRangeTensorOffsets) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_FALSE(file.ok());
    EXPECT_EQ(file.status().code(), StatusCode::kOutOfRange);
}

TEST(ModelLoader_HfSafetensorsFileTest, RejectsShapeByteSizeMismatch) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_FALSE(file.ok());
    EXPECT_EQ(file.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfSafetensorsFileTest, FindReturnsNullptrForUnknownTensor) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);

    ASSERT_TRUE(file.ok()) << file.status().ToString();
    EXPECT_EQ(file->Find("missing"), nullptr);
}

TEST(ModelLoader_HfSafetensorsFileTest, KeepsStorageAliveThroughViews) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{5.0f, 6.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    RawWeightView view;
    {
        const auto file = HfSafetensorsFile::Open(path);
        ASSERT_TRUE(file.ok()) << file.status().ToString();
        const auto* entry = file->Find("weight");
        ASSERT_NE(entry, nullptr);
        view = entry->view;
    }

    ASSERT_TRUE(view.IsValid());
    ASSERT_NE(view.storage, nullptr);
    EXPECT_FLOAT_EQ(ReadFloat(view.data), 5.0f);
    EXPECT_FLOAT_EQ(ReadFloat(view.data + sizeof(float)), 6.0f);
}

TEST(ModelLoader_HfSafetensorsFileTest, KeepsMappedViewAliveAfterFileIsRemoved) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{7.0f, 8.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.Path(),
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    const auto file = HfSafetensorsFile::Open(path);
    ASSERT_TRUE(file.ok()) << file.status().ToString();
    const auto* entry = file->Find("weight");
    ASSERT_NE(entry, nullptr);
    const RawWeightView view = entry->view;

    std::error_code error;
    ASSERT_TRUE(std::filesystem::remove(path, error));
    ASSERT_FALSE(error) << error.message();

    ASSERT_TRUE(view.IsValid());
    EXPECT_FLOAT_EQ(ReadFloat(view.data), 7.0f);
    EXPECT_FLOAT_EQ(ReadFloat(view.data + sizeof(float)), 8.0f);
}

}// namespace
}// namespace aethermind
