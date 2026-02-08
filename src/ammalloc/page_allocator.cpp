//
// Created by richard on 2/7/26.
//
#include "ammalloc/page_allocator.h"
#include "spdlog/spdlog.h"

#include <sys/mman.h>

namespace aethermind {

namespace {

void* AllocWithRetry(size_t size, int flags) {
    for (size_t i = 0; i < PageConfig::MAX_ALLOC_RETRIES; ++i) {
        if (void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
            ptr != MAP_FAILED) {
            return ptr;
        }

        if (errno != ENOMEM) {
            spdlog::error("mmap fatal error: errno={}", errno);
            break;
        }

        spdlog::warn("mmap ENOMEM for size {}, retry {}/{}...",
                     size, i + 1, PageConfig::MAX_ALLOC_RETRIES);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return MAP_FAILED;
}

void ApplyHugePageHint(void* ptr, size_t size) {
    if (madvise(ptr, size, MADV_HUGEPAGE) != 0) {
        spdlog::debug("madvise MADV_HUGEPAGE failed (expected on non-THP systems).");
    }

    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        if (madvise(ptr, size, MADV_WILLNEED) != 0) {
            spdlog::warn("madvise MADV_WILLNEED failed.");
        }
    }
}

void* AllocNormalPage(size_t size, PageAllocatorStats& stats) {
    stats.normal_alloc_count.fetch_add(1, std::memory_order_relaxed);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (RuntimeConfig::GetInstance().UseMapPopulate()) {
        flags |= MAP_POPULATE;
    }

    void* ptr = AllocWithRetry(size, flags);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }

    stats.normal_alloc_success.fetch_add(1, std::memory_order_relaxed);
    stats.normal_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    return ptr;
}

void* AllocHugePageRobust(size_t size, PageAllocatorStats& stats) {
    size_t alloc_size = size + SystemConfig::HUGE_PAGE_SIZE;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = AllocWithRetry(alloc_size, flags);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }

    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned_addr = (addr + SystemConfig::HUGE_PAGE_SIZE - 1) &
                                   ~(SystemConfig::HUGE_PAGE_SIZE - 1);
    size_t waste = 0;
    const size_t head_gap = aligned_addr - addr;
    if (head_gap > 0) {
        munmap(ptr, head_gap);
        waste += head_gap;
    }

    if (const size_t tail_gap = alloc_size - head_gap - size; tail_gap > 0) {
        munmap(reinterpret_cast<void*>(aligned_addr + size), tail_gap);
        waste += tail_gap;
    }

    stats.huge_align_waste_bytes.fetch_add(waste, std::memory_order_relaxed);

    auto* res = reinterpret_cast<void*>(aligned_addr);
    ApplyHugePageHint(res, size);
    stats.huge_alloc_success.fetch_add(1, std::memory_order_relaxed);
    stats.huge_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    return res;
}

/**
     * @brief 乐观的大页分配策略 [优化1]
     * 策略：
     * 1. 先尝试直接申请 size 大小（赌它刚好对齐）。
     * 2. 如果不对齐，munmap 掉，再走 "Over-allocate" 流程。
     * 收益：在内存碎片较少或 OS 激进 THP 时，避免了 2MB 的 VMA 浪费和额外的 munmap 调用。
     */
void* AllocHugePage(size_t size, PageAllocatorStats& stats) {
    stats.huge_alloc_count.fetch_add(1, std::memory_order_relaxed);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    void* ptr = AllocWithRetry(size, flags);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }

    auto addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & (SystemConfig::HUGE_PAGE_SIZE - 1)) == 0) {
        ApplyHugePageHint(ptr, size);
        stats.huge_alloc_success.fetch_add(1, std::memory_order_relaxed);
        stats.huge_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
        return ptr;
    }

    munmap(ptr, size);
    return AllocHugePageRobust(size, stats);
}

}// namespace


void* PageAllocator::SystemAlloc(size_t page_num) {
    const size_t size = page_num << SystemConfig::PAGE_SHIFT;

    // clang-format off
    if (size < (SystemConfig::HUGE_PAGE_SIZE >> 1)) AM_LIKELY {
        void* ptr = AllocNormalPage(size, stats_);
        if (!ptr) {
            stats_.alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
        }
        return ptr;
    }
    // clang-format on

    auto* ptr = AllocHugePage(size, stats_);
    if (ptr == nullptr) {
        ptr = AllocNormalPage(size, stats_);
    }

    if (ptr == nullptr) {
        stats_.alloc_failed_count.fetch_add(1, std::memory_order_relaxed);
    }
    return ptr;
}

void PageAllocator::SystemFree(void* ptr, size_t page_num) {
    if (!ptr || page_num == 0) {
        return;
    }

    const size_t size = page_num << SystemConfig::PAGE_SHIFT;
    stats_.free_count.fetch_add(1, std::memory_order_relaxed);
    stats_.free_bytes.fetch_add(size, std::memory_order_relaxed);
    munmap(ptr, size);
}

}// namespace aethermind