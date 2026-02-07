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

void PageAllocator::ApplyHugePageHint(void* ptr, size_t size) {
    if (madvise(ptr, size, MADV_HUGEPAGE) != 0) {
        spdlog::debug("madvise MADV_HUGEPAGE failed (expected on non-THP systems).");
    }

    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        if (madvise(ptr, size, MADV_WILLNEED) != 0) {
            spdlog::warn("madvise MADV_WILLNEED failed.");
        }
    }
}

void* PageAllocator::AllocHugePageRobust(size_t size) {
    size_t alloc_size = size + SystemConfig::HUGE_PAGE_SIZE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        spdlog::error("AllocHugePageRobust: mmap failed for size {}: {}",
                      alloc_size, strerror(errno));
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

    auto* res = reinterpret_cast<void*>(aligned_addr);
    ApplyHugePageHint(res, size);
    return res;
}

/**
     * @brief 乐观的大页分配策略 [优化1]
     * 策略：
     * 1. 先尝试直接申请 size 大小（赌它刚好对齐）。
     * 2. 如果不对齐，munmap 掉，再走 "Over-allocate" 流程。
     * 收益：在内存碎片较少或 OS 激进 THP 时，避免了 2MB 的 VMA 浪费和额外的 munmap 调用。
     */
void* PageAllocator::AllocHugePage(size_t size) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }

    auto addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & (SystemConfig::HUGE_PAGE_SIZE - 1)) == 0) {
        ApplyHugePageHint(ptr, size);
        return ptr;
    }

    munmap(ptr, size);
    return AllocHugePageRobust(size);
}

}// namespace aethermind