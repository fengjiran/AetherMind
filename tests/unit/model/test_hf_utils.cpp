#include "aethermind/model/formats/hf/hf_utils.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <string_view>

namespace {

using namespace aethermind;
using namespace aethermind::hf;

struct DTypeMapping {
    std::string_view dtype_text;
    DataType expected;
};

class ParseSafetensorsDTypeTest
    : public testing::TestWithParam<DTypeMapping> {};

TEST_P(ParseSafetensorsDTypeTest, ParsesSupportedDType) {
    const auto dtype = ParseSafetensorsDType(GetParam().dtype_text);

    ASSERT_TRUE(dtype.ok()) << dtype.status().ToString();
    EXPECT_EQ(*dtype, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
        ModelLoader_HfUtilsTest, ParseSafetensorsDTypeTest,
        testing::Values(
                DTypeMapping{"F16", DataType::Float(16)},
                DTypeMapping{"BF16", DataType::BFloat(16)},
                DTypeMapping{"F32", DataType::Float32()},
                DTypeMapping{"F64", DataType::Float(64)},
                DTypeMapping{"I8", DataType::Int(8)},
                DTypeMapping{"I16", DataType::Int(16)},
                DTypeMapping{"I32", DataType::Int(32)},
                DTypeMapping{"I64", DataType::Int(64)},
                DTypeMapping{"U8", DataType::UInt(8)},
                DTypeMapping{"U16", DataType::UInt(16)},
                DTypeMapping{"U32", DataType::UInt(32)},
                DTypeMapping{"U64", DataType::UInt(64)},
                DTypeMapping{"BOOL", DataType::Bool()},
                DTypeMapping{"F8_E5M2", DataType::Float8E5M2()},
                DTypeMapping{"F8_E4M3", DataType::Float8E4M3()}));

TEST(ModelLoader_HfUtilsTest, RejectsUnknownDType) {
    const auto dtype = ParseSafetensorsDType("F4");

    ASSERT_FALSE(dtype.ok());
    EXPECT_EQ(dtype.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(dtype.status().message().find("Unsupported safetensors dtype"),
              std::string::npos);
}

TEST(ModelLoader_HfUtilsTest, RejectsLowercaseDTypeName) {
    const auto dtype = ParseSafetensorsDType("f32");

    ASSERT_FALSE(dtype.ok());
    EXPECT_EQ(dtype.status().code(), StatusCode::kInvalidArgument);
}

TEST(ModelLoader_HfUtilsTest, ReadFileTextReadsContent) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "config.json";
    WriteTextFile(path, "{\"model_type\": \"llama\"}");

    const auto text = ReadFileText(path);

    ASSERT_TRUE(text.ok()) << text.status().ToString();
    EXPECT_EQ(*text, "{\"model_type\": \"llama\"}");
}

TEST(ModelLoader_HfUtilsTest, ReadFileTextMissingFile) {
    TempDirectory temp_dir;

    const auto text = ReadFileText(temp_dir.path() / "missing.json");

    ASSERT_FALSE(text.ok());
    EXPECT_EQ(text.status().code(), StatusCode::kNotFound);
}

TEST(ModelLoader_HfUtilsTest, ReadFileTextRejectsDirectory) {
    TempDirectory temp_dir;

    const auto text = ReadFileText(temp_dir.path());

    ASSERT_FALSE(text.ok());
    EXPECT_EQ(text.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(text.status().message().find("not a regular file"),
              std::string::npos);
}

TEST(ModelLoader_HfUtilsTest, ReadFileTextRejectsSymlink) {
    TempDirectory temp_dir;
    const auto target = temp_dir.path() / "config.json";
    WriteTextFile(target, "{}");
    const auto link = temp_dir.path() / "config_link.json";
    std::filesystem::create_symlink(target, link);

    const auto text = ReadFileText(link);

    ASSERT_FALSE(text.ok());
    EXPECT_EQ(text.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(text.status().message().find("must not be a symlink"),
              std::string::npos);
}

TEST(ModelLoader_HfUtilsTest, ReadFileTextRejectsOversizedFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "big.json";
    {
        std::ofstream stream(path, std::ios::binary);
        ASSERT_TRUE(stream.is_open()) << path.string();
        constexpr size_t kMaxTextFileSize = 16ULL * 1024 * 1024;
        const std::vector<char> chunk(1024 * 1024, 'x');
        size_t written = 0;
        while (written <= kMaxTextFileSize) {
            stream.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            written += chunk.size();
        }
    }

    const auto text = ReadFileText(path);

    ASSERT_FALSE(text.ok());
    EXPECT_EQ(text.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(text.status().message().find("exceeds maximum size"),
              std::string::npos);
}

TEST(ModelLoader_HfUtilsTest, RejectExistingPathIfSymlinkPassesRegularFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "config.json";
    WriteTextFile(path, "{}");

    const auto status = RejectExistingPathIfSymlink(path, "config.json");

    ASSERT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfUtilsTest, RejectExistingPathIfSymlinkPassesMissingPath) {
    TempDirectory temp_dir;

    const auto status =
            RejectExistingPathIfSymlink(temp_dir.path() / "missing.json", "config.json");

    ASSERT_TRUE(status.ok()) << status.ToString();
}

TEST(ModelLoader_HfUtilsTest, RejectExistingPathIfSymlinkRejectsFileSymlink) {
    TempDirectory temp_dir;
    const auto target = temp_dir.path() / "config.json";
    WriteTextFile(target, "{}");
    const auto link = temp_dir.path() / "config_link.json";
    std::filesystem::create_symlink(target, link);

    const auto status = RejectExistingPathIfSymlink(link, "config.json");

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("must not be a symlink"), std::string::npos);
}

TEST(ModelLoader_HfUtilsTest, RejectExistingPathIfSymlinkRejectsDirectorySymlink) {
    TempDirectory temp_dir;
    const auto target = temp_dir.path() / "target_dir";
    std::filesystem::create_directory(target);
    const auto link = temp_dir.path() / "dir_link";
    std::filesystem::create_symlink(target, link);

    const auto status = RejectExistingPathIfSymlink(link, "model dir");

    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

} // namespace