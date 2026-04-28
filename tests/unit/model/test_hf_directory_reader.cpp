#include "aethermind/model/formats/hf/hf_directory_reader.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

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

TEST(HfDirectoryReaderTest, DiscoverSingleFileLayout) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto layout = HfDirectoryReader::DiscoverLayout(temp_dir.path());

    ASSERT_TRUE(layout.ok()) << layout.status().ToString();
    EXPECT_TRUE(layout->IsSingleFile());
    EXPECT_EQ(layout->layout, HfDirectoryLayout::kSingleSafetensors);
    EXPECT_EQ(layout->config_path, temp_dir.path() / "config.json");
    EXPECT_EQ(layout->safetensors_path, temp_dir.path() / "model.safetensors");
    EXPECT_TRUE(layout->safetensors_index_path.empty());
}

TEST(HfDirectoryReaderTest, RejectsMissingConfigJson) {
    TempDirectory temp_dir;
    WriteBinaryFile(temp_dir.path() / "model.safetensors");

    const auto layout = HfDirectoryReader::DiscoverLayout(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kNotFound);
}

TEST(HfDirectoryReaderTest, RejectsMissingSafetensorsFile) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");

    const auto layout = HfDirectoryReader::DiscoverLayout(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kNotFound);
}

TEST(HfDirectoryReaderTest, RejectsNonDirectoryPath) {
    TempDirectory temp_dir;
    const auto file_path = temp_dir.path() / "config.json";
    WriteTextFile(file_path, "{}");

    const auto layout = HfDirectoryReader::DiscoverLayout(file_path);

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(HfDirectoryReaderTest, RejectsConflictingSingleAndShardedLayout) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteBinaryFile(temp_dir.path() / "model.safetensors");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json", "{}");

    const auto layout = HfDirectoryReader::DiscoverLayout(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kFailedPrecondition);
}

TEST(HfDirectoryReaderTest, ReportsShardedLayoutAsUnimplemented) {
    TempDirectory temp_dir;
    WriteTextFile(temp_dir.path() / "config.json", "{}");
    WriteTextFile(temp_dir.path() / "model.safetensors.index.json", "{}");

    const auto layout = HfDirectoryReader::DiscoverLayout(temp_dir.path());

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kUnimplemented);
}

}// namespace
}// namespace aethermind
