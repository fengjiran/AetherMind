//
// Created by richard on 5/12/26.
//

#include "aethermind/base/mmap_file.h"

#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace aethermind {

namespace {

std::string ErrnoMessage(const char* operation, const std::filesystem::path& path, int error_number) {
    return std::string(operation) + " failed for file '" + path.string() + "': " +
           std::error_code(error_number, std::generic_category()).message();
}

}// namespace

MemoryMappedFile::~MemoryMappedFile() {
    if (data_ != nullptr) {
        munmap(data_, size_);
    }
}

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

MemoryMappedFile& MemoryMappedFile::operator=(MemoryMappedFile&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            munmap(data_, size_);
        }
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

StatusOr<MemoryMappedFile> MemoryMappedFile::Map(const std::filesystem::path& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status::Internal(ErrnoMessage("open", path, errno));
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        const int error_number = errno;
        close(fd);
        return Status::Internal(ErrnoMessage("fstat", path, error_number));
    }

    if (!S_ISREG(sb.st_mode)) {
        close(fd);
        return Status::InvalidArgument("Path is not a regular file: " + path.string());
    }

    if (sb.st_size <= 0) {
        close(fd);
        return Status::InvalidArgument("File is empty: " + path.string());
    }

    const auto file_size = static_cast<std::uintmax_t>(sb.st_size);
    if (file_size > std::numeric_limits<size_t>::max()) {
        close(fd);
        return Status::InvalidArgument("File is too large to map into this process: " + path.string());
    }

    const auto size = static_cast<size_t>(file_size);
    void* data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (data == MAP_FAILED) {
        const int error_number = errno;
        close(fd);
        return Status::Internal(ErrnoMessage("mmap", path, error_number));
    }

    close(fd);// mmap 成功后，fd 可以安全关闭

    return MemoryMappedFile(data, size);
}

}// namespace aethermind
