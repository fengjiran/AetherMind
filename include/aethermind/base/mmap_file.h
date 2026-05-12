//
// Created by richard on 5/12/26.
//

#ifndef AETHERMIND_BASE_MMAP_FILE_H
#define AETHERMIND_BASE_MMAP_FILE_H

#include "aethermind/base/status.h"

#include <cstddef>
#include <filesystem>
#include <span>

namespace aethermind {

class MemoryMappedFile {
public:
    MemoryMappedFile() noexcept = default;
    ~MemoryMappedFile();

    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    AM_NODISCARD static StatusOr<MemoryMappedFile> Map(const std::filesystem::path& path);

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

}// namespace aethermind

#endif// AETHERMIND_BASE_MMAP_FILE_H
