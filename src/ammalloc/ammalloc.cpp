//
// Created by richard on 2/21/26.
//

#include "ammalloc/ammalloc.h"
#include "ammalloc/page_allocator.h"
#include "ammalloc/page_cache.h"
#include "ammalloc/thread_cache.h"

namespace {

using namespace aethermind;

thread_local ThreadCache* pTLSThreadCache = nullptr;
thread_local bool g_ThreadCacheAlreadyDestructed = false;

ThreadCache* CreateThreadCache() {
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

void ReleaseThreadCache(ThreadCache* tc) {
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

thread_local ThreadCacheCleaner tc_cleaner;

}// namespace

namespace aethermind {

void* am_malloc(size_t size) {
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

void am_free(void* ptr) {
    if (!ptr) {
        return;
    }

    auto* span = PageMap::GetSpan(ptr);
    if (!span) {
        return;
    }

    auto size = span->obj_size;
    // If the size is 0, it means the span is allocated from the system(big object).
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