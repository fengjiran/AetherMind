#include "aethermind/model/formats/hf/hf_safetensors_index.h"

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

    AM_NODISCARD const std::filesystem::path& path() const noexcept {
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

TEST(HfSafetensorsIndexTest, LoadsSingleTensorFile) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})" ,
            raw_bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    ASSERT_EQ(index->Entries().size(), 1U);
    const auto* entry = index->Find("weight");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->dtype, DataType::Float32());
    EXPECT_EQ(entry->shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(entry->data_offset_begin, 0U);
    EXPECT_EQ(entry->data_offset_end, 16U);
    EXPECT_TRUE(entry->view.IsValid());
    const auto* values = reinterpret_cast<const float*>(entry->view.data);
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[3], 4.0f);
}

TEST(HfSafetensorsIndexTest, LoadsMultipleTensorEntries) {
    TempDirectory temp_dir;
    const auto w1_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto w2_bytes = FloatArrayToBytes(std::array<float, 2>{3.0f, 4.0f});
    std::vector<std::byte> raw_bytes;
    raw_bytes.insert(raw_bytes.end(), w1_bytes.begin(), w1_bytes.end());
    raw_bytes.insert(raw_bytes.end(), w2_bytes.begin(), w2_bytes.end());

    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"w1":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"w2":{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]}})" ,
            raw_bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_EQ(index->Entries().size(), 2U);
    ASSERT_NE(index->Find("w1"), nullptr);
    ASSERT_NE(index->Find("w2"), nullptr);
}

TEST(HfSafetensorsIndexTest, RejectsInvalidHeaderLength) {
    TempDirectory temp_dir;
    std::vector<std::byte> bytes;
    const auto prefix = EncodeLittleEndianU64(1024);
    bytes.insert(bytes.end(), prefix.begin(), prefix.end());
    bytes.push_back(static_cast<std::byte>('{'));
    const auto path = temp_dir.path() / "model.safetensors";
    WriteRawFile(path, bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfSafetensorsIndexTest, RejectsMalformedHeaderJson) {
    TempDirectory temp_dir;
    const auto header = R"({"weight":)";
    const auto path = WriteSafetensorsFile(temp_dir.path(), header, {});

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfSafetensorsIndexTest, RejectsUnknownDType) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"weight":{"dtype":"F128","shape":[2],"data_offsets":[0,8]}})" ,
            raw_bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfSafetensorsIndexTest, RejectsOutOfRangeTensorOffsets) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})" ,
            raw_bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kOutOfRange);
}

TEST(HfSafetensorsIndexTest, RejectsShapeByteSizeMismatch) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}})" ,
            raw_bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_FALSE(index.ok());
    EXPECT_EQ(index.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfSafetensorsIndexTest, FindReturnsNullptrForUnknownTensor) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})" ,
            raw_bytes);

    const auto index = HfSafetensorsIndex::LoadSingleFile(path);

    ASSERT_TRUE(index.ok()) << index.status().ToString();
    EXPECT_EQ(index->Find("missing"), nullptr);
}

TEST(HfSafetensorsIndexTest, KeepsBackingAliveThroughViews) {
    TempDirectory temp_dir;
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{5.0f, 6.0f});
    const auto path = WriteSafetensorsFile(
            temp_dir.path(),
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})" ,
            raw_bytes);

    RawTensorView view;
    {
        const auto index = HfSafetensorsIndex::LoadSingleFile(path);
        ASSERT_TRUE(index.ok()) << index.status().ToString();
        const auto* entry = index->Find("weight");
        ASSERT_NE(entry, nullptr);
        view = entry->view;
    }

    ASSERT_TRUE(view.IsValid());
    ASSERT_NE(view.backing, nullptr);
    const auto* values = reinterpret_cast<const float*>(view.data);
    EXPECT_FLOAT_EQ(values[0], 5.0f);
    EXPECT_FLOAT_EQ(values[1], 6.0f);
}

}// namespace
}// namespace aethermind
