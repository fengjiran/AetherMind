//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
#define AETHERMIND_MALLOC_PAGE_ALLOCATOR_H

#include "ammalloc/config.h"

#include <atomic>
#include <sys/mman.h>

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

    std::atomic<size_t> alloc_failed_count{0};
};

class PageAllocator {
public:
    static void* SystemAlloc(size_t page_num) {
        const size_t size = page_num << SystemConfig::PAGE_SHIFT;
        // clang-format off
        if (size < (SystemConfig::HUGE_PAGE_SIZE >> 1)) AM_LIKELY {
            return AllocNormalPage(size);
        }
        // clang-format on

        return AllocHugePage(size);
    }

    static void SystemFree(void* ptr, size_t page_num) {
        if (!ptr || page_num == 0) {
            return;
        }

        const size_t size = page_num << SystemConfig::PAGE_SHIFT;
        munmap(ptr, size);
    }

private:
    static void* AllocNormalPage(size_t size);

    static void ApplyHugePageHint(void* ptr, size_t size);
    static void* AllocHugePage(size_t size);
    static void* AllocHugePageRobust(size_t size);
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
