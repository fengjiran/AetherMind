#ifndef AETHERMIND_BASE_MMAP_FILE_H
#define AETHERMIND_BASE_MMAP_FILE_H

/// @file mmap_file.h
/// @brief Read-only memory-mapped file wrapper with POSIX madvise support.
#include "aethermind/base/status.h"

#include <cstddef>
#include <filesystem>
#include <span>

namespace aethermind {

/// @brief Read-only memory-mapped file backed by mmap(2).
///
/// Move-only RAII wrapper. The mapping is released on destruction or
/// move-assignment. Provides byte-level access to the mapped region and
/// an optional posix_madvise hint for OS read-ahead optimization.
/// @note Not thread-safe. Concurrent access requires external synchronization.
class MemoryMappedFile {
public:
    /// @brief OS-level access-pattern hints forwarded to posix_madvise.
    enum class Advice {
        kNormal,
        kRandom,
        kSequential,
        kWillNeed,
        kDontNeed,
    };

    MemoryMappedFile() noexcept = default;
    ~MemoryMappedFile();

    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    /// @brief Maps an entire regular file as read-only.
    /// @param path Filesystem path to the file to map.
    /// @return A valid MemoryMappedFile on success, or an error Status
    ///         (kNotFound, kPermissionDenied, kInvalidArgument, kInternal).
    AM_NODISCARD static StatusOr<MemoryMappedFile> Map(const std::filesystem::path& path);

    /// @brief Applies an OS-level access-pattern hint to the entire mapped range.
    /// @param advice The access-pattern hint to apply.
    /// @return Ok on success; kInvalidArgument if the mapping is invalid,
    ///         kInternal if posix_madvise fails.
    /// @note This is an optimization hint only; callers must not rely on it
    ///       for correctness.
    AM_NODISCARD Status Advise(Advice advice) const;

    AM_NODISCARD const void* data() const noexcept {
        return data_;
    }

    AM_NODISCARD const std::byte* ByteData() const noexcept {
        return static_cast<const std::byte*>(data_);
    }

    AM_NODISCARD std::span<const std::byte> Bytes() const noexcept {
        return {ByteData(), size_};
    }

    AM_NODISCARD size_t size() const noexcept {
        return size_;
    }

    AM_NODISCARD bool valid() const noexcept {
        return data_ != nullptr;
    }

private:
    MemoryMappedFile(void* data, size_t size) noexcept : data_(data), size_(size) {}

    void* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace aethermind

#endif // AETHERMIND_BASE_MMAP_FILE_H
