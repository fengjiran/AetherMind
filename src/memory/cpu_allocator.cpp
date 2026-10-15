//
// Created by 赵丹 on 25-6-25.
//

#include "aethermind/memory/cpu_allocator.h"
#include "aethermind/base/env.h"
#include "alignment.h"
#include "utils/logging.h"
#include "utils/memory_utils.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace aethermind {

namespace {

void CpuBufferDeleter(void*, void* ptr) noexcept {
    free(ptr);
}

bool IsThpAllocEnabled() {
    static const bool value = [] {
        const auto env = check_env("THP_MEM_ALLOC_ENABLE");
        return env.has_value() ? env.value() != 0 : false;
    }();
    return value;
}

size_t ThpAlignment() {
    static const size_t value = [] {
        const long pagesize = sysconf(_SC_PAGESIZE);
        return pagesize > 0 ? static_cast<size_t>(pagesize) : gPagesize;
    }();
    return value;
}

bool ThpApplies(const CPUAllocatorPolicy& policy, size_t bytes) {
    return policy.allow_thp && IsThpAllocEnabled() && bytes >= gAlloc_threshold_thp;
}

} // namespace

CPUAllocator::CPUAllocator(Device device,
                           CPUAllocatorPolicy policy,
                           std::shared_ptr<AllocationState> state)
    : Allocator(std::move(state)), device_(device), policy_(policy) {}

StatusOr<MemoryHandle> CPUAllocator::AllocateStorage(const AllocationRequest& request) {
    // A zero-byte Buffer still needs a distinct, freeable pointer.
    const size_t bytes = request.nbytes == 0 ? 1 : request.nbytes;

    const size_t policy_default =
            policy_.default_alignment != 0 ? policy_.default_alignment : gAlignment;
    const size_t minimum = request.alignment == 0 ? policy_default : request.alignment;
    size_t alignment = std::max(minimum, sizeof(void*));

    const bool thp = ThpApplies(policy_, bytes);
    if (thp) {
        alignment = std::max(alignment, ThpAlignment());
    }

    void* data = nullptr;
    const int rc = posix_memalign(&data, alignment, bytes);
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

    // Advisory logging may allocate, so storage must already have an owner.
    MemoryHandle storage(data, nullptr, &CpuBufferDeleter, device_, alignment);

    if (thp) {
#ifdef __linux__
        if (madvise(data, bytes, MADV_HUGEPAGE) != 0) {
            spdlog::warn("thp madvise for HUGEPAGE failed");
        }
#endif
    }

    if (policy_.junk_fill) {
        FillMemoryJunk(data, bytes);
    }

    return storage;
}

Device CPUAllocator::device() const noexcept {
    return device_;
}

std::unique_ptr<Allocator> CPUAllocatorProvider::CreateAllocator(
        Device device, std::shared_ptr<AllocationState> state) {
    AM_CHECK(device.type() == DeviceType::kCPU, "CPUAllocatorProvider only supports CPU devices.");
    return std::make_unique<CPUAllocator>(device, CPUAllocatorPolicy{},
                                          std::move(state));
}

} // namespace aethermind
