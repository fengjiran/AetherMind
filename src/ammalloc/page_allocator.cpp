//
// Created by richard on 2/7/26.
//
#include "ammalloc/page_allocator.h"
#include "spdlog/spdlog.h"

#include <cstring>

namespace aethermind {

void* PageAllocator::AllocNormalPage(size_t size) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        flags |= MAP_POPULATE;
    }

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        spdlog::error("mmap failed for size {}: {}", size, strerror(errno));
        return nullptr;
    }

    return ptr;
}

void* PageAllocator::AllocHugePage(size_t size) {
    size_t alloc_size = size + SystemConfig::HUGE_PAGE_SIZE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        spdlog::error("mmap failed for size {}: {}", alloc_size, strerror(errno));
        return nullptr;
    }

    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned_addr = (addr + SystemConfig::HUGE_PAGE_SIZE - 1) &
                                   ~(SystemConfig::HUGE_PAGE_SIZE - 1);
    const size_t head_gap = aligned_addr - addr;
    if (head_gap > 0) {
        munmap(ptr, head_gap);
    }

    if (const size_t tail_gap = alloc_size - head_gap - size; tail_gap > 0) {
        munmap(reinterpret_cast<void*>(aligned_addr + size), tail_gap);
    }

    madvise(reinterpret_cast<void*>(aligned_addr), size, MADV_HUGEPAGE);
    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        madvise(reinterpret_cast<void*>(aligned_addr), size, MADV_WILLNEED);
    }

    return reinterpret_cast<void*>(aligned_addr);
}

}// namespace aethermind