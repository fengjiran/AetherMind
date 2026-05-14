//
// Created by richard on 5/12/26.
//

#include "aethermind/base/mmap_file.h"
#include "utils/logging.h"

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

class ScopedFd {
public:
    explicit ScopedFd(int fd) noexcept : fd_(fd) {}
    ~ScopedFd() {
        (void) Close();
    }

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    AM_NODISCARD int get() const noexcept {
        return fd_;
    }

    AM_NODISCARD int Close() noexcept {
        if (fd_ < 0) {
            return 0;
        }

        const int fd = fd_;
        fd_ = -1;
        return close(fd);
    }

private:
    int fd_ = -1;
};

std::string ErrnoMessage(const char* operation, const std::filesystem::path& path, int error_number) {
    return std::string(operation) + " failed for file '" + path.string() + "': " +
           std::error_code(error_number, std::generic_category()).message();
}

Status CloseBeforeReturn(ScopedFd& fd, const std::filesystem::path& path, const Status& status) {
    if (fd.Close() == 0) {
        return status;
    }

    return Status::Internal(ErrnoMessage("close", path, errno) +
                            "; original status: " + status.ToString());
}

int ToPosixAdvice(MemoryMappedFile::Advice advice) noexcept {
    switch (advice) {
        case MemoryMappedFile::Advice::kNormal:
            return POSIX_MADV_NORMAL;
        case MemoryMappedFile::Advice::kRandom:
            return POSIX_MADV_RANDOM;
        case MemoryMappedFile::Advice::kSequential:
            return POSIX_MADV_SEQUENTIAL;
        case MemoryMappedFile::Advice::kWillNeed:
            return POSIX_MADV_WILLNEED;
        case MemoryMappedFile::Advice::kDontNeed:
            return POSIX_MADV_DONTNEED;
    }
    return POSIX_MADV_NORMAL;
}

}// namespace

MemoryMappedFile::~MemoryMappedFile() {
    if (data_ != nullptr) {
        const int ret = munmap(data_, size_);
        AM_DCHECK(ret == 0, "munmap failed");
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
            const int ret = munmap(data_, size_);
            AM_DCHECK(ret == 0, "munmap failed");
        }
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Status MemoryMappedFile::Advise(Advice advice) const {
    if (data_ == nullptr || size_ == 0) {
        return Status::InvalidArgument("Cannot advise an invalid memory mapping");
    }

    if (const int error_number = posix_madvise(data_, size_, ToPosixAdvice(advice)); error_number != 0) {
        return Status::Internal(std::string("posix_madvise failed: ") +
                                std::error_code(error_number, std::generic_category()).message());
    }
    return Status::Ok();
}

StatusOr<MemoryMappedFile> MemoryMappedFile::Map(const std::filesystem::path& path) {
    ScopedFd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.get() < 0) {
        if (errno == ENOENT) {
            return Status::NotFound(ErrnoMessage("open", path, errno));
        }

        if (errno == EACCES || errno == EPERM) {
            return Status::PermissionDenied(ErrnoMessage("open", path, errno));
        }

        return Status::Internal(ErrnoMessage("open", path, errno));
    }

    struct stat sb;
    if (fstat(fd.get(), &sb) == -1) {
        const int error_number = errno;
        return CloseBeforeReturn(fd, path,
                                 Status::Internal(ErrnoMessage("fstat", path, error_number)));
    }

    if (!S_ISREG(sb.st_mode)) {
        return CloseBeforeReturn(fd, path,
                                 Status::InvalidArgument("Path is not a regular file: " + path.string()));
    }

    if (sb.st_size <= 0) {
        return CloseBeforeReturn(fd, path,
                                 Status::InvalidArgument("File is empty: " + path.string()));
    }

    const auto file_size = static_cast<std::uintmax_t>(sb.st_size);
    if (file_size > std::numeric_limits<size_t>::max()) {
        return CloseBeforeReturn(fd, path,
                                 Status::InvalidArgument("File is too large to map into this process: " + path.string()));
    }

    const auto size = static_cast<size_t>(file_size);
    void* data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);

    if (data == MAP_FAILED) {
        const int error_number = errno;
        return CloseBeforeReturn(fd, path,
                                 Status::Internal(ErrnoMessage("mmap", path, error_number)));
    }

    // Once a read-only mmap succeeds, the kernel holds the file reference needed by the mapping.
    // close() only releases this descriptor, so a late close error is intentionally non-fatal.
    (void) fd.Close();

    return MemoryMappedFile(data, size);
}

}// namespace aethermind
