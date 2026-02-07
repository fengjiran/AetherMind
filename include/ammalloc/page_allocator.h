//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
#define AETHERMIND_MALLOC_PAGE_ALLOCATOR_H

#include "ammalloc/config.h"

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
    static void* AllocNormalPage(size_t size);

    static void* AllocHugePage(size_t size);
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
