//
// Created by 赵丹 on 25-6-25.
//

#include "aethermind/memory/cpu_allocator.h"

#include "aethermind/base/alignment.h"
#include "aethermind/base/env.h"
#include "alignment.h"
#include "utils/memory_utils.h"

#include <cerrno>
#include <cstdlib>
#include <new>
#include <spdlog/spdlog.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace aethermind {

namespace {

void cpu_buffer_deleter(void*, void* ptr) noexcept {
    free_cpu(ptr);
}

bool is_thp_alloc_enabled() {
    static bool value = [&] {
        auto env = check_env("THP_MEM_ALLOC_ENABLE");
        return env.has_value() ? env.value() : 0;
    }();
    return value;
}

bool is_thp_alloc(size_t nbytes) {
    return is_thp_alloc_enabled() && nbytes >= gAlloc_threshold_thp;
}

} // namespace

size_t get_alignment(size_t nbytes) {
    static const auto pagesize = sysconf(_SC_PAGESIZE);
    const size_t thp_alignment = pagesize < 0 ? gPagesize : pagesize;
    return is_thp_alloc(nbytes) ? thp_alignment : gAlignment;
}

void* alloc_cpu(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }

    AM_CHECK(static_cast<ptrdiff_t>(nbytes) >= 0, "the nbytes is seems a negative number.");

    void* data = nullptr;
    int err = posix_memalign(&data, get_alignment(nbytes), nbytes);
    if (err == ENOMEM) {
        throw std::bad_alloc();
    }
    AM_CHECK(err == 0, "Try allocate {} bytes memory failed.", nbytes);

    if (is_thp_alloc(nbytes)) {
#ifdef __linux__
        int ret = madvise(data, nbytes, MADV_HUGEPAGE);
        if (ret != 0) {
            spdlog::warn("thp madvise for HUGEPAGE failed");
        }
#endif
    }

    FillMemoryJunk(data, nbytes);

    return data;
}

void free_cpu(void* data) {
    free(data);
}

StatusOr<Buffer> TryAllocateAlignedCpuBuffer(size_t nbytes,
                                             size_t alignment) noexcept try {
    if (alignment == 0 || !IsValidAlignment(alignment) ||
        alignment % sizeof(void*) != 0) {
        return Status::InvalidArgument("CPU buffer alignment must be a non-zero "
                                       "power of two and a multiple of sizeof(void*)");
    }

    void* data = nullptr;
    const int rc = posix_memalign(&data, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc == EINVAL) {
        return Status::InvalidArgument(
                "CPU buffer alignment is invalid for posix_memalign");
    }

    if (rc == ENOMEM) {
        return Status::ResourceExhausted({});
    }

    if (rc != 0) {
        return Status::Internal(
                "posix_memalign failed with an unexpected error");
    }

    if (data == nullptr) {
        return Status::Internal(
                "posix_memalign succeeded with a null pointer");
    }

    return Buffer{nbytes,
                  MemoryHandle(data, nullptr, &cpu_buffer_deleter,
                               Device::CPU(), alignment)};
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

Buffer CPUAllocator::Allocate(size_t nbytes) {
    const size_t bytes_to_allocate = nbytes == 0 ? 1 : nbytes;
    void* data = alloc_cpu(bytes_to_allocate);
    return {nbytes, MemoryHandle(data, nullptr, &cpu_buffer_deleter,
                                 device_, get_alignment(bytes_to_allocate))};
}

Device CPUAllocator::device() const noexcept {
    return device_;
}

std::unique_ptr<Allocator> CPUAllocatorProvider::CreateAllocator(Device device) {
    AM_CHECK(device.type() == DeviceType::kCPU, "CPUAllocatorProvider only supports CPU devices.");
    return std::make_unique<CPUAllocator>(device);
}

} // namespace aethermind
