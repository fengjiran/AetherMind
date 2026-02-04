//
// Created by richard on 2/4/26.
//
module;

#include "macros.h"
#include "utils/logging.h"

#include <mutex>

export module ammalloc;

import ammemory_pool;

namespace aethermind {

thread_local ThreadCache* pTLSThreadCache = nullptr;

static ThreadCache* CreateThreadCache() {
    static std::mutex tc_init_mtx;
    std::lock_guard<std::mutex> lock(tc_init_mtx);
    if (pTLSThreadCache) {
        return pTLSThreadCache;
    }

    constexpr auto tc_size = sizeof(ThreadCache);
    constexpr auto page_num = (tc_size + MagicConstants::PAGE_SIZE - 1) >> MagicConstants::PAGE_SHIFT;
    void* ptr = PageAllocator::Allocate(page_num);
    return new (ptr) ThreadCache;
}

static void ReleaseThreadCache(ThreadCache* tc) {
    if (!tc) {
        return;
    }

    tc->~ThreadCache();
    constexpr auto tc_size = sizeof(ThreadCache);
    constexpr auto page_num = (tc_size + MagicConstants::PAGE_SIZE - 1) >> MagicConstants::PAGE_SHIFT;
    PageAllocator::Release(tc, page_num);
}

struct ThreadCacheCleaner {
    ~ThreadCacheCleaner() {
        if (pTLSThreadCache) {
            pTLSThreadCache->ReleaseAll();
            ReleaseThreadCache(pTLSThreadCache);
            pTLSThreadCache = nullptr;
        }
    }
};

thread_local ThreadCacheCleaner tc_cleaner;

export void* am_malloc(size_t size) {
    if (size > MagicConstants::MAX_TC_SIZE) {
        const auto align_size = (size + MagicConstants::PAGE_SIZE - 1) & ~(MagicConstants::PAGE_SIZE - 1);
        const size_t page_num = align_size >> MagicConstants::PAGE_SHIFT;
        const auto* span = PageCache::GetInstance().AllocSpan(page_num, 0);
        if (!span) {
            return nullptr;
        }

        return span->GetStartAddr();
    }

    // clang-format off
    if (!pTLSThreadCache) AM_UNLIKELY {
        pTLSThreadCache = CreateThreadCache();
    }
    // clang-format on

    return pTLSThreadCache->Allocate(size);
}

export void am_free(void* ptr) {
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
    if (!pTLSThreadCache) AM_UNLIKELY{
        pTLSThreadCache = CreateThreadCache();
    }
    // clang-format on

    pTLSThreadCache->Deallocate(ptr, size);
}

}// namespace aethermind
