//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
#define AETHERMIND_MALLOC_PAGE_ALLOCATOR_H

#include "ammalloc/config.h"
#include "spdlog/spdlog.h"

#include <cstring>
#include <sys/mman.h>

namespace aethermind {

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
    static void* AllocNormalPage(size_t size) {
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

    static void* AllocHugePage(size_t size) {
        size_t alloc_size = size + SystemConfig::HUGE_PAGE_SIZE;
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (ptr == MAP_FAILED) {
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
        return reinterpret_cast<void*>(aligned_addr);
    }
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
