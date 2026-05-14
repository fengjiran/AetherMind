#include "aethermind/base/mmap_file.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
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
                ("aethermind_mmap_test_" + unique_id);
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    AM_NODISCARD const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_{};
};

void WriteFile(const std::filesystem::path& path,
               std::string_view content) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << "Failed to create test file: " << path.string();
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
}

void WriteFile(const std::filesystem::path& path,
               const std::vector<std::byte>& bytes) {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open()) << "Failed to create test file: " << path.string();
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
}

TEST(MemoryMappedFile, MapsSmallFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "small.bin";
    WriteFile(path, "Hello, mmap!");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->valid());
    EXPECT_EQ(result->size(), 12U);
    EXPECT_NE(result->data(), nullptr);
    EXPECT_NE(result->ByteData(), nullptr);

    const auto* data = static_cast<const char*>(result->data());
    EXPECT_EQ(std::string(data, result->size()), "Hello, mmap!");
}

TEST(MemoryMappedFile, MapsBinaryFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "binary.bin";
    std::vector<std::byte> bytes(256);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::byte>(i);
    }
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 256U);
    const auto span = result->Bytes();
    EXPECT_EQ(span.size(), 256U);
    for (size_t i = 0; i < span.size(); ++i) {
        EXPECT_EQ(span[i], static_cast<std::byte>(i))
                << "Mismatch at byte " << i;
    }
}

TEST(MemoryMappedFile, MapsSingleByteFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "one.bin";
    WriteFile(path, std::string(1, '\x42'));

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 1U);
    EXPECT_EQ(result->ByteData()[0], static_cast<std::byte>(0x42));
}

TEST(MemoryMappedFile, MapsPageAlignedFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "page.bin";
    const size_t page_size = 4096;
    std::vector<std::byte> bytes(page_size, static_cast<std::byte>(0xAB));
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), page_size);
    EXPECT_EQ(result->Bytes()[0], static_cast<std::byte>(0xAB));
    EXPECT_EQ(result->Bytes()[page_size - 1], static_cast<std::byte>(0xAB));
}

TEST(MemoryMappedFile, MapsMultiPageFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "multipage.bin";
    const size_t size = 4096 * 4 + 123;
    std::vector<std::byte> bytes(size, static_cast<std::byte>(0xCD));
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), size);
    EXPECT_TRUE(std::all_of(result->Bytes().begin(), result->Bytes().end(),
                            [](std::byte b) { return b == static_cast<std::byte>(0xCD); }));
}

TEST(MemoryMappedFile, ByteDataMatchesData) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "consistency.bin";
    WriteFile(path, "testdata");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->data(), static_cast<const void*>(result->ByteData()));
}

TEST(MemoryMappedFile, BytesSpanMatchesByteDataAndSize) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "span_check.bin";
    WriteFile(path, "span_data_here");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto span = result->Bytes();
    EXPECT_EQ(span.data(), result->ByteData());
    EXPECT_EQ(span.size(), result->size());
}

TEST(MemoryMappedFile, DefaultConstructedIsInvalid) {
    MemoryMappedFile mmap;

    EXPECT_FALSE(mmap.valid());
    EXPECT_EQ(mmap.data(), nullptr);
    EXPECT_EQ(mmap.ByteData(), nullptr);
    EXPECT_EQ(mmap.size(), 0U);
    EXPECT_TRUE(mmap.Bytes().empty());
}

TEST(MemoryMappedFile, AdviseRejectsInvalidMapping) {
    MemoryMappedFile mmap;

    const Status status = mmap.Advise(MemoryMappedFile::Advice::kNormal);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MemoryMappedFile, AdvisesMappedFileAccessPattern) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "advise.bin";
    WriteFile(path, "advise_data");

    auto mmap = MemoryMappedFile::Map(path);
    ASSERT_TRUE(mmap.ok()) << mmap.status().ToString();

    EXPECT_TRUE(mmap->Advise(MemoryMappedFile::Advice::kNormal).ok());
    EXPECT_TRUE(mmap->Advise(MemoryMappedFile::Advice::kRandom).ok());
    EXPECT_TRUE(mmap->Advise(MemoryMappedFile::Advice::kSequential).ok());
    EXPECT_TRUE(mmap->Advise(MemoryMappedFile::Advice::kWillNeed).ok());
    EXPECT_TRUE(mmap->Advise(MemoryMappedFile::Advice::kDontNeed).ok());
}

TEST(MemoryMappedFile, MoveConstructionTransfersOwnership) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "move_ctor.bin";
    WriteFile(path, "move_me");

    auto original = MemoryMappedFile::Map(path);
    ASSERT_TRUE(original.ok()) << original.status().ToString();
    const void* original_data = original->data();
    size_t original_size = original->size();

    MemoryMappedFile moved(std::move(*original));

    EXPECT_TRUE(moved.valid());
    EXPECT_EQ(moved.data(), original_data);
    EXPECT_EQ(moved.size(), original_size);
    EXPECT_FALSE(original->valid());
    EXPECT_EQ(original->data(), nullptr);
    EXPECT_EQ(original->size(), 0U);

    const auto* chars = static_cast<const char*>(moved.data());
    EXPECT_EQ(std::string(chars, moved.size()), "move_me");
}

TEST(MemoryMappedFile, MoveAssignmentTransfersOwnership) {
    TempDirectory temp_dir;
    const auto path1 = temp_dir.path() / "move_assign1.bin";
    const auto path2 = temp_dir.path() / "move_assign2.bin";
    WriteFile(path1, "file_one_data");
    WriteFile(path2, "file_two_data_here");

    auto mmap1 = MemoryMappedFile::Map(path1);
    auto mmap2 = MemoryMappedFile::Map(path2);
    ASSERT_TRUE(mmap1.ok()) << mmap1.status().ToString();
    ASSERT_TRUE(mmap2.ok()) << mmap2.status().ToString();

    const void* mmap2_data = mmap2->data();
    size_t mmap2_size = mmap2->size();

    mmap1 = std::move(*mmap2);

    EXPECT_EQ(mmap1->data(), mmap2_data);
    EXPECT_EQ(mmap1->size(), mmap2_size);
    EXPECT_FALSE(mmap2->valid());
    EXPECT_EQ(mmap2->data(), nullptr);
    EXPECT_EQ(mmap2->size(), 0U);

    const auto* chars = static_cast<const char*>(mmap1->data());
    EXPECT_EQ(std::string(chars, mmap1->size()), "file_two_data_here");
}

TEST(MemoryMappedFile, MoveAssignmentReleasesPreviousMapping) {
    TempDirectory temp_dir;
    const auto path1 = temp_dir.path() / "prev_mapping.bin";
    const auto path2 = temp_dir.path() / "new_mapping.bin";
    WriteFile(path1, "old_data_contents");
    WriteFile(path2, "new_data_contents");

    auto result1 = MemoryMappedFile::Map(path1);
    auto result2 = MemoryMappedFile::Map(path2);
    ASSERT_TRUE(result1.ok()) << result1.status().ToString();
    ASSERT_TRUE(result2.ok()) << result2.status().ToString();

    const void* old_data = result1->data();
    EXPECT_NE(old_data, nullptr);

    *result1 = std::move(*result2);

    EXPECT_NE(result1->data(), old_data);
}

TEST(MemoryMappedFile, SelfMoveAssignmentIsNoop) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "self_assign.bin";
    WriteFile(path, "self_assign_data");

    auto result = MemoryMappedFile::Map(path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const void* original_data = result->data();
    size_t original_size = result->size();

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
#endif
    *result = std::move(*result);
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    EXPECT_EQ(result->data(), original_data);
    EXPECT_EQ(result->size(), original_size);
    EXPECT_TRUE(result->valid());
}

TEST(MemoryMappedFile, DestructorReleasesMapping) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "dtor_test.bin";
    WriteFile(path, "destroy_me");

    void* captured_data = nullptr;
    {
        auto result = MemoryMappedFile::Map(path);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        captured_data = const_cast<void*>(result->data());
        EXPECT_NE(captured_data, nullptr);
    }

    EXPECT_NE(captured_data, nullptr);
}

TEST(MemoryMappedFile, MapRejectsNonExistentPath) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "does_not_exist.bin";

    const auto result = MemoryMappedFile::Map(path);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kNotFound);
}

TEST(MemoryMappedFile, MapRejectsNonExistentDirectory) {
    const auto path = std::filesystem::path("/no/such/directory/file.bin");

    const auto result = MemoryMappedFile::Map(path);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kNotFound);
}

TEST(MemoryMappedFile, MapRejectsDirectoryAsPath) {
    TempDirectory temp_dir;

    const auto result = MemoryMappedFile::Map(temp_dir.path());

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(MemoryMappedFile, MapRejectsEmptyFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "empty.bin";
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open());

    const auto result = MemoryMappedFile::Map(path);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(MemoryMappedFile, MapRejectsUnreadableFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "noperm.bin";
    WriteFile(path, "secret_data");

    std::filesystem::permissions(path,
                                 std::filesystem::perms::none,
                                 std::filesystem::perm_options::replace);

    const auto result = MemoryMappedFile::Map(path);

    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.status().code() == StatusCode::kPermissionDenied ||
                result.status().code() == StatusCode::kInternal)
            << "Actual status: " << result.status().ToString();

    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
}

TEST(MemoryMappedFile, MapRejectsNonRegularFileSymlink) {
    TempDirectory temp_dir;
    const auto link = temp_dir.path() / "dir_link";
    std::error_code ec;
    std::filesystem::create_directory_symlink(temp_dir.path(), link, ec);
    if (ec) {
        GTEST_SKIP() << "Cannot create directory symlink for testing: " << ec.message();
    }

    const auto result = MemoryMappedFile::Map(link);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(MemoryMappedFile, MapAbsolutePath) {
    TempDirectory temp_dir;
    const auto path = std::filesystem::canonical(temp_dir.path()) / "absolute.bin";
    WriteFile(path, "absolute_path_data");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 18U);
}

TEST(MemoryMappedFile, MapRelativePath) {
    TempDirectory temp_dir;
    const auto rel_path = temp_dir.path() / "subdir";
    std::filesystem::create_directories(rel_path);
    const auto file_path = rel_path / "relative.bin";
    WriteFile(file_path, "relative_data");

    const auto result = MemoryMappedFile::Map(file_path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 13U);
}

TEST(MemoryMappedFile, MapPathWithSpaces) {
    TempDirectory temp_dir;
    const auto subdir = temp_dir.path() / "path with spaces";
    std::filesystem::create_directories(subdir);
    const auto path = subdir / "file name.bin";
    WriteFile(path, "spaces_data");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 11U);
}

TEST(MemoryMappedFile, MapPathWithUnicode) {
    TempDirectory temp_dir;
    const auto subdir = temp_dir.path() / "路径_тест_パス";
    std::filesystem::create_directories(subdir);
    const auto path = subdir / "数据.bin";
    WriteFile(path, "unicode_path_data");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 17U);
}

TEST(MemoryMappedFile, MapPathWithSpecialChars) {
    TempDirectory temp_dir;
    const auto subdir = temp_dir.path() / "dir-with.special+chars";
    std::filesystem::create_directories(subdir);
    const auto path = subdir / "file[1].bin";
    WriteFile(path, "special_chars");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 13U);
}

TEST(MemoryMappedFile, MapDeeplyNestedPath) {
    TempDirectory temp_dir;
    auto deep_path = temp_dir.path();
    for (int i = 0; i < 10; ++i) {
        deep_path /= "level" + std::to_string(i);
    }
    std::filesystem::create_directories(deep_path);
    const auto file_path = deep_path / "deep.bin";
    WriteFile(file_path, "deeply_nested");

    const auto result = MemoryMappedFile::Map(file_path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 13U);
}

TEST(MemoryMappedFile, MapLongPathName) {
    TempDirectory temp_dir;
    std::string long_name(200, 'A');
    const auto path = temp_dir.path() / (long_name + ".bin");
    WriteFile(path, "long_name_data");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 14U);
}

TEST(MemoryMappedFile, MapSameFileTwice) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "shared.bin";
    WriteFile(path, "shared_content_data");

    auto result1 = MemoryMappedFile::Map(path);
    auto result2 = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result1.ok()) << result1.status().ToString();
    ASSERT_TRUE(result2.ok()) << result2.status().ToString();
    EXPECT_EQ(result1->size(), result2->size());

    const auto* data1 = static_cast<const char*>(result1->data());
    const auto* data2 = static_cast<const char*>(result2->data());
    EXPECT_EQ(std::string(data1, result1->size()),
              std::string(data2, result2->size()));
}

TEST(MemoryMappedFile, MapSameFileSequentially) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "sequential.bin";
    WriteFile(path, "sequential_data");

    {
        auto result1 = MemoryMappedFile::Map(path);
        ASSERT_TRUE(result1.ok()) << result1.status().ToString();
        EXPECT_EQ(result1->size(), 15U);
    }

    {
        auto result2 = MemoryMappedFile::Map(path);
        ASSERT_TRUE(result2.ok()) << result2.status().ToString();
        EXPECT_EQ(result2->size(), 15U);
    }
}

TEST(MemoryMappedFile, MapMultipleFilesConcurrently) {
    TempDirectory temp_dir;
    constexpr size_t kFileCount = 8;
    std::vector<StatusOr<MemoryMappedFile>> mappings;
    mappings.reserve(kFileCount);

    for (size_t i = 0; i < kFileCount; ++i) {
        const auto path = temp_dir.path() / ("concurrent_" + std::to_string(i) + ".bin");
        WriteFile(path, "data_" + std::to_string(i));
        mappings.push_back(MemoryMappedFile::Map(path));
    }

    for (size_t i = 0; i < kFileCount; ++i) {
        ASSERT_TRUE(mappings[i].ok()) << mappings[i].status().ToString();
        EXPECT_TRUE(mappings[i]->valid());
        const auto* chars = static_cast<const char*>(mappings[i]->data());
        EXPECT_EQ(std::string(chars, mappings[i]->size()),
                  "data_" + std::to_string(i));
    }
}

TEST(MemoryMappedFile, MapFileWithExactContentSize) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "exact.bin";
    const std::string content(4097, 'X');
    WriteFile(path, content);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 4097U);
    EXPECT_EQ(static_cast<const char*>(result->data())[0], 'X');
    EXPECT_EQ(static_cast<const char*>(result->data())[4096], 'X');
}

TEST(MemoryMappedFile, MapLargerFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "larger.bin";
    constexpr size_t kSize = 1024 * 1024;
    std::vector<std::byte> bytes(kSize);
    for (size_t i = 0; i < kSize; ++i) {
        bytes[i] = static_cast<std::byte>(i % 256);
    }
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), kSize);
    for (size_t i = 0; i < kSize; ++i) {
        EXPECT_EQ(result->ByteData()[i], static_cast<std::byte>(i % 256))
                << "Mismatch at byte " << i;
    }
}

TEST(MemoryMappedFile, ReadAfterMoveOriginalInvalid) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "move_read.bin";
    WriteFile(path, "read_after_move");

    auto original = MemoryMappedFile::Map(path);
    ASSERT_TRUE(original.ok()) << original.status().ToString();

    MemoryMappedFile moved(std::move(*original));

    EXPECT_FALSE(original->valid());
    EXPECT_EQ(original->ByteData(), nullptr);
    EXPECT_EQ(original->size(), 0U);
    EXPECT_TRUE(original->Bytes().empty());

    const auto* chars = static_cast<const char*>(moved.data());
    EXPECT_EQ(std::string(chars, moved.size()), "read_after_move");
}

TEST(MemoryMappedFile, MapEmptyPathString) {
    const auto result = MemoryMappedFile::Map(std::filesystem::path(""));

    EXPECT_FALSE(result.ok());
}

TEST(MemoryMappedFile, MapDotPath) {
    TempDirectory temp_dir;
    const auto original = std::filesystem::current_path();
    std::filesystem::current_path(temp_dir.path());
    const auto file_path = temp_dir.path() / "cwd_file.bin";
    WriteFile(file_path, "cwd_data");

    const auto dot_result = MemoryMappedFile::Map(std::filesystem::path("."));
    EXPECT_FALSE(dot_result.ok());
    EXPECT_EQ(dot_result.status().code(), StatusCode::kInvalidArgument);

    std::filesystem::current_path(original);
}

TEST(MemoryMappedFile, MapSymlinkToFile) {
    TempDirectory temp_dir;
    const auto target = temp_dir.path() / "target.bin";
    WriteFile(target, "symlink_target_data");
    const auto link = temp_dir.path() / "link.bin";

    std::error_code ec;
    std::filesystem::create_symlink(target, link, ec);
    if (ec) {
        GTEST_SKIP() << "Cannot create symlink for testing: " << ec.message();
    }

    const auto result = MemoryMappedFile::Map(link);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 19U);
    const auto* chars = static_cast<const char*>(result->data());
    EXPECT_EQ(std::string(chars, result->size()), "symlink_target_data");
}

TEST(MemoryMappedFile, MapBrokenSymlink) {
    TempDirectory temp_dir;
    const auto link = temp_dir.path() / "broken_link.bin";

    std::error_code ec;
    std::filesystem::create_symlink("/nonexistent_target_for_test", link, ec);
    if (ec) {
        GTEST_SKIP() << "Cannot create broken symlink for testing: " << ec.message();
    }

    const auto result = MemoryMappedFile::Map(link);

    EXPECT_FALSE(result.ok());
}

TEST(MemoryMappedFile, MapHardlinkToFile) {
    TempDirectory temp_dir;
    const auto target = temp_dir.path() / "hardlink_target.bin";
    WriteFile(target, "hardlink_target_data");
    const auto link = temp_dir.path() / "hardlink.bin";

    std::error_code ec;
    std::filesystem::create_hard_link(target, link, ec);
    if (ec) {
        GTEST_SKIP() << "Cannot create hard link for testing: " << ec.message();
    }

    const auto result = MemoryMappedFile::Map(link);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 20U);
}

TEST(MemoryMappedFile, MappedDataIsReadConsistent) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "consistent.bin";
    const std::string content = "consistent_data_for_mmap";
    WriteFile(path, content);

    auto result = MemoryMappedFile::Map(path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();

    for (size_t i = 0; i < 3; ++i) {
        const auto* chars = static_cast<const char*>(result->data());
        EXPECT_EQ(std::string(chars, result->size()), content)
                << "Inconsistent read on attempt " << (i + 1);
    }
}

TEST(MemoryMappedFile, MapOnlyReadableFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "readonly.bin";
    WriteFile(path, "readonly_data");

    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read,
                                 std::filesystem::perm_options::replace);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 13U);

    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
}

TEST(MemoryMappedFile, ValidReturnsTrueAfterSuccessfulMap) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "valid.bin";
    WriteFile(path, "x");

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->valid());
}

TEST(MemoryMappedFile, ValidReturnsFalseForDefaultConstructed) {
    MemoryMappedFile mmap;
    EXPECT_FALSE(mmap.valid());
}

TEST(MemoryMappedFile, ValidReturnsFalseAfterMove) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "moved_valid.bin";
    WriteFile(path, "y");

    auto original = MemoryMappedFile::Map(path);
    ASSERT_TRUE(original.ok()) << original.status().ToString();
    EXPECT_TRUE(original->valid());

    MemoryMappedFile moved(std::move(*original));
    EXPECT_FALSE(original->valid());
    EXPECT_TRUE(moved.valid());
}

TEST(MemoryMappedFile, MapDevNull) {
    const auto result = MemoryMappedFile::Map(std::filesystem::path("/dev/null"));

    EXPECT_FALSE(result.ok());
}

TEST(MemoryMappedFile, MapDevZero) {
    const auto result = MemoryMappedFile::Map(std::filesystem::path("/dev/zero"));

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(MemoryMappedFile, MapZeroByteFile) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "zero_byte.bin";
    {
        std::ofstream stream(path, std::ios::binary);
        ASSERT_TRUE(stream.is_open());
    }

    const auto result = MemoryMappedFile::Map(path);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(MemoryMappedFile, MoveAssignmentFromDefaultConstructed) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "assign_from_default.bin";
    WriteFile(path, "target_data");

    auto target_result = MemoryMappedFile::Map(path);
    ASSERT_TRUE(target_result.ok()) << target_result.status().ToString();
    MemoryMappedFile target = std::move(*target_result);
    EXPECT_TRUE(target.valid());

    MemoryMappedFile empty;
    target = std::move(empty);

    EXPECT_FALSE(target.valid());
    EXPECT_EQ(target.data(), nullptr);
    EXPECT_EQ(target.size(), 0U);
}

TEST(MemoryMappedFile, MoveAssignToDefaultConstructed) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "assign_to_default.bin";
    WriteFile(path, "source_data");

    auto source_result = MemoryMappedFile::Map(path);
    ASSERT_TRUE(source_result.ok()) << source_result.status().ToString();
    const void* source_data = source_result->data();

    MemoryMappedFile target;
    target = std::move(*source_result);

    EXPECT_TRUE(target.valid());
    EXPECT_EQ(target.data(), source_data);
    EXPECT_EQ(target.size(), 11U);
    EXPECT_FALSE(source_result->valid());
}

TEST(MemoryMappedFile, BytesSpanDataMatchesFileContent) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "span_content.bin";
    const uint32_t values[] = {0xDEADBEEF, 0xCAFEBABE, 0x12345678};
    std::vector<std::byte> bytes(sizeof(values));
    std::memcpy(bytes.data(), values, sizeof(values));
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto span = result->Bytes();
    ASSERT_EQ(span.size(), sizeof(values));

    uint32_t read_values[3];
    std::memcpy(read_values, span.data(), sizeof(values));
    EXPECT_EQ(read_values[0], 0xDEADBEEFu);
    EXPECT_EQ(read_values[1], 0xCAFEBABEu);
    EXPECT_EQ(read_values[2], 0x12345678u);
}

TEST(MemoryMappedFile, MapFileWithNullBytes) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "nulls.bin";
    std::vector<std::byte> bytes = {
            static_cast<std::byte>(0x00),
            static_cast<std::byte>(0xFF),
            static_cast<std::byte>(0x00),
            static_cast<std::byte>(0x01),
            static_cast<std::byte>(0x00),
    };
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 5U);
    EXPECT_EQ(result->ByteData()[0], static_cast<std::byte>(0x00));
    EXPECT_EQ(result->ByteData()[1], static_cast<std::byte>(0xFF));
    EXPECT_EQ(result->ByteData()[4], static_cast<std::byte>(0x00));
}

TEST(MemoryMappedFile, MapFileThenFileDeleted) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "deleted.bin";
    WriteFile(path, "will_be_deleted_data");

    auto result = MemoryMappedFile::Map(path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const void* mapped_data = result->data();

    std::error_code ec;
    std::filesystem::remove(path, ec);

    const auto* chars = static_cast<const char*>(mapped_data);
    EXPECT_EQ(std::string(chars, result->size()), "will_be_deleted_data");
}

TEST(MemoryMappedFile, MapFileWithAllByteValues) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "all_bytes.bin";
    std::vector<std::byte> bytes(256);
    for (unsigned i = 0; i < 256; ++i) {
        bytes[i] = static_cast<std::byte>(i);
    }
    WriteFile(path, bytes);

    const auto result = MemoryMappedFile::Map(path);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 256U);
    for (unsigned i = 0; i < 256; ++i) {
        EXPECT_EQ(result->ByteData()[i], static_cast<std::byte>(i))
                << "Byte value mismatch at offset " << i;
    }
}

TEST(MemoryMappedFile, RepeatedMapUnmapCycle) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "cycle.bin";
    WriteFile(path, "cycle_test_data");

    for (int i = 0; i < 5; ++i) {
        auto result = MemoryMappedFile::Map(path);
        ASSERT_TRUE(result.ok()) << result.status().ToString() << " on iteration " << i;
        EXPECT_EQ(result->size(), 15U);
        const auto* chars = static_cast<const char*>(result->data());
        EXPECT_EQ(std::string(chars, 15), "cycle_test_data");
    }
}

TEST(MemoryMappedFile, MmapPointerIsPageAligned) {
    TempDirectory temp_dir;
    const auto path = temp_dir.path() / "aligned.bin";
    WriteFile(path, "alignment_check");

    const auto result = MemoryMappedFile::Map(path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();

    const auto addr = reinterpret_cast<uintptr_t>(result->data());
    EXPECT_EQ(addr % 4096, 0U)
            << "mmap returned non-page-aligned address: " << addr;
}

}// namespace
}// namespace aethermind
