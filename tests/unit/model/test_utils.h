#ifndef AETHERMIND_MODEL_TEST_UTILS_H
#define AETHERMIND_MODEL_TEST_UTILS_H

#include "macros.h"

#include <array>
#include <chrono>
#include <cstddef>
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

class TempDirectory {
public:
    TempDirectory() {
        static size_t counter = 0;
        const auto unique_id = std::to_string(
                                       std::chrono::steady_clock::now().time_since_epoch().count()) +
                               "_" + std::to_string(counter++);
        path_ = std::filesystem::temp_directory_path() /
                ("aethermind_test_" + unique_id);
        std::filesystem::create_directories(path_);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    TempDirectory(TempDirectory&&) = delete;
    TempDirectory& operator=(TempDirectory&&) = delete;

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    AM_NODISCARD const std::filesystem::path& path() const noexcept {// NOLINT(readability-identifier-naming)
        return path_;
    }

private:
    std::filesystem::path path_{};
};

inline void WriteTextFile(const std::filesystem::path& path,
                          std::string_view content) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    stream << content;
}

inline void WriteBinaryFile(const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    constexpr char kZero = 0;
    stream.write(&kZero, 1);
}

inline std::vector<std::byte> ToBytes(std::string_view text) {
    std::vector<std::byte> bytes(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

inline std::array<std::byte, sizeof(uint64_t)> EncodeLittleEndianU64(uint64_t value) {
    std::array<std::byte, sizeof(uint64_t)> bytes{};
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        bytes[i] = static_cast<std::byte>((value >> (8U * i)) & 0xFFU);
    }
    return bytes;
}

inline std::vector<std::byte> FloatArrayToBytes(std::span<const float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

inline float ReadFloat(const std::byte* data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(float));
    return value;
}

inline void WriteRawFile(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << path.string();
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
}

inline void WriteSafetensorsFile(const std::filesystem::path& path,
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

template<size_t N>
void WriteSafetensorsFileWithFloats(const std::filesystem::path& path,
                                           std::string_view header_json,
                                           const std::array<float, N>& values) {
    const auto raw_bytes = FloatArrayToBytes(values);
    WriteSafetensorsFile(path, header_json, raw_bytes);
}

}// namespace aethermind

#endif