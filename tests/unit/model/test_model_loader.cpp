#include "aethermind/model/model_loader.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/model/model_instance.h"

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
                ("aethermind_model_loader_" + unique_id);
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

void WriteTextFile(const std::filesystem::path& path, std::string_view content) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    stream << content;
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

TEST(ModelLoaderTest, ValidSingleFileDirectoryReachesModelInstanceBoundary) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.Path() / "config.json", "{}");
    const auto raw_bytes = FloatArrayToBytes(std::array<float, 2>{1.0f, 2.0f});
    WriteSafetensorsFile(
            temp_dir.Path() / "model.safetensors",
            R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            raw_bytes);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.Path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kUnimplemented);
    EXPECT_NE(model.status().message().find("reader tensor table"), std::string::npos);
}

TEST(ModelLoaderTest, PropagatesSafetensorsArtifactError) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.Path() / "config.json", "{}");
    const auto prefix = EncodeLittleEndianU64(1024);
    WriteRawFile(temp_dir.Path() / "model.safetensors", prefix);

    CpuBackend backend;
    KernelRegistry registry;
    const auto model = ModelLoader::Load(ModelLoadOptions{.model_dir = temp_dir.Path()}, backend, registry);

    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
