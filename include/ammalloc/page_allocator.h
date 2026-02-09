//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
#define AETHERMIND_MALLOC_PAGE_ALLOCATOR_H

#include "ammalloc/config.h"

#include <atomic>

namespace aethermind {

struct PageAllocatorStats {
    // 基础分配统计
    std::atomic<size_t> normal_alloc_count{0};    // 普通页分配请求数
    std::atomic<size_t> normal_alloc_success{0};  // 普通页分配成功数
    std::atomic<size_t> normal_alloc_bytes{0};    // 普通页总分配字节数
    std::atomic<size_t> huge_alloc_count{0};      // 大页分配请求数
    std::atomic<size_t> huge_alloc_success{0};    // 大页分配成功数
    std::atomic<size_t> huge_alloc_bytes{0};      // 大页总分配字节数
    std::atomic<size_t> huge_align_waste_bytes{0};// 大页对齐浪费的内存字节数
    // 释放统计
    std::atomic<size_t> free_count{0};
    std::atomic<size_t> free_bytes{0};

    // 错误统计
    std::atomic<size_t> alloc_failed_count{0};           // 最终分配失败总数
    std::atomic<size_t> huge_fallback_to_normal_count{0};// 大页降级到普通页的次数
    std::atomic<size_t> normal_alloc_failed_count{0};    // 普通页分配失败次数
    std::atomic<size_t> huge_alloc_failed_count{0};      // 大页分配失败次数
    std::atomic<size_t> munmap_failed_count{0};          // munmap失败次数
    std::atomic<size_t> madvise_failed_count{0};         // madvise失败次数
    std::atomic<size_t> mmap_enomem_count{0};            // mmap ENOMEM失败次数
    std::atomic<size_t> mmap_other_error_count{0};       // mmap其他错误次数
};

class PageAllocator {
public:
    static const PageAllocatorStats& GetStats() {
        return stats_;
    }

    static void ResetStats();

    static void* SystemAlloc(size_t page_num);
    static void SystemFree(void* ptr, size_t page_num);

private:
    inline static PageAllocatorStats stats_;

    static void* AllocWithRetry(size_t size, int flags);
    static void ApplyHugePageHint(void* ptr, size_t size);
    static void* AllocNormalPage(size_t size, bool is_fallback = false);
    static void* AllocHugePageFallback(size_t size);
    static void* AllocHugePage(size_t size);
    static bool SafeMunmap(void* ptr, size_t size);
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
