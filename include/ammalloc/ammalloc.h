//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_AMMALLOC_H
#define AETHERMIND_MALLOC_AMMALLOC_H

#include "ammalloc/memory_pool.h"
#include "ammalloc/page_allocator.h"

namespace aethermind {

inline thread_local ThreadCache* pTLSThreadCache = nullptr;
inline thread_local bool g_ThreadCacheAlreadyDestructed = false;

static ThreadCache* CreateThreadCache() {
    if (g_ThreadCacheAlreadyDestructed) {
        return nullptr;
    }

    static std::mutex tc_init_mtx;
    std::lock_guard<std::mutex> lock(tc_init_mtx);
    if (pTLSThreadCache) {
        return pTLSThreadCache;
    }

    constexpr auto tc_size = sizeof(ThreadCache);
    constexpr auto page_num = (tc_size + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
    void* ptr = PageAllocator::SystemAlloc(page_num);
    return new (ptr) ThreadCache;
}

static void ReleaseThreadCache(ThreadCache* tc) {
    if (!tc) {
        return;
    }

    tc->~ThreadCache();
    constexpr auto tc_size = sizeof(ThreadCache);
    constexpr auto page_num = (tc_size + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
    PageAllocator::SystemFree(tc, page_num);
}

struct ThreadCacheCleaner {
    ~ThreadCacheCleaner() {
        g_ThreadCacheAlreadyDestructed = true;
        if (pTLSThreadCache) {
            pTLSThreadCache->ReleaseAll();
            ReleaseThreadCache(pTLSThreadCache);
            pTLSThreadCache = nullptr;
        }
    }
};

inline thread_local ThreadCacheCleaner tc_cleaner;

inline void* am_malloc(size_t size) {
    if (size > SizeConfig::MAX_TC_SIZE) {
        const auto align_size = (size + SystemConfig::PAGE_SIZE - 1) & ~(SystemConfig::PAGE_SIZE - 1);
        const size_t page_num = align_size >> SystemConfig::PAGE_SHIFT;
        const auto* span = PageCache::GetInstance().AllocSpan(page_num, 0);
        if (!span) {
            return nullptr;
        }

        return span->GetStartAddr();
    }

    // clang-format off
    if (!pTLSThreadCache) AM_UNLIKELY {
        pTLSThreadCache = CreateThreadCache();
        if (!pTLSThreadCache) {
            return nullptr;
        }
    }
    // clang-format on

    return pTLSThreadCache->Allocate(size);
}

inline void am_free(void* ptr) {
    if (!ptr) {
        return;
    }

    auto* span = PageMap::GetSpan(ptr);
    if (!span) {
        return;
    }

    auto size = span->obj_size;
    if (size == 0) {
        PageCache::GetInstance().ReleaseSpan(span);
        return;
    }

    // clang-format off
    if (!pTLSThreadCache) AM_UNLIKELY {
        pTLSThreadCache = CreateThreadCache();
        if (!pTLSThreadCache) {
            static_cast<FreeBlock*>(ptr)->next = nullptr;
            CentralCache::GetInstance().ReleaseListToSpans(ptr, size);
            return;
        }
    }
    // clang-format on

    pTLSThreadCache->Deallocate(ptr, size);
}


}// namespace aethermind

#endif//AETHERMIND_MALLOC_AMMALLOC_H
