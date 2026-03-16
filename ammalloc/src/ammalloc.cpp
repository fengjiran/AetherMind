//
// Created by richard on 2/21/26.
//

#include "ammalloc/ammalloc.h"
#include "ammalloc/config.h"
#include "ammalloc/page_allocator.h"
#include "ammalloc/page_cache.h"
#include "ammalloc/page_heap_scavenger.h"
#include "ammalloc/thread_cache.h"
#include <atomic>
#include <cstddef>

namespace {

using namespace aethermind;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((tls_model("initial-exec")))
#endif
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

// Minimal, safe and lock-free background scavenger thread starter
void EnsureScavengerStarted() noexcept {
    if (!RuntimeConfig::GetInstance().EnableScavenger()) {
        return;
    }

    static std::atomic<bool> started{false};

    // Most cases are fast path, just check if the thread is already started
    // clang-format off
    if (!started.load(std::memory_order_acquire)) AM_UNLIKELY {
        bool expected = false;
        if (started.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel)) {
            try {
                PageHeapScavenger::GetInstance().Start();
            } catch (const std::exception& e) {
                // If thread creation fails (e.g., system resources exhausted),
                // log the error and swallow the exception.
                // At this point 'started' remains true, so no retries will be attempted.
                // The memory pool can still function without a background scavenger thread,
                // but it will just won't proactively reduce RSS.
                spdlog::warn("Failed to start PageHeapScavenger: {}. Continuing without background GC.", e.what());
            } catch (...) {
                spdlog::error("Unknown exception while starting PageHeapScavenger.");
            }
        }
    }
    // clang-format on
}

AM_NOINLINE void* am_malloc_slow_path(size_t size) {
    // Note: Thread creation may take a few milliseconds.
    // This is acceptable because:
    // 1. It's on the slow path (already paying for mmap/munmap)
    // 2. It only happens once per process lifetime
    // 3. The alternative (early thread creation) impacts process startup time
    EnsureScavengerStarted();

    if (size > SizeConfig::MAX_TC_SIZE) {
        const auto align_size = (size + SystemConfig::PAGE_SIZE - 1) & ~(SystemConfig::PAGE_SIZE - 1);
        const size_t page_num = align_size >> SystemConfig::PAGE_SHIFT;
        const auto* span = PageCache::GetInstance().AllocSpan(page_num);
        if (!span) {
            return nullptr;
        }

        return span->GetStartAddr();
    }

    if (!pTLSThreadCache) {
        pTLSThreadCache = CreateThreadCache();
        if (!pTLSThreadCache) {
            return nullptr;
        }
    }

    return pTLSThreadCache->Allocate(size);
}

AM_NOINLINE void am_free_slow_path(void* ptr, Span* span, size_t size) {
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

}// namespace

namespace aethermind {

void* am_malloc(size_t size) {
    // TLS variable is read only once.
    auto* tc = pTLSThreadCache;
    // clang-format off
    if (size > SizeConfig::MAX_TC_SIZE || tc == nullptr) AM_UNLIKELY {
        return am_malloc_slow_path(size);
    }
    // clang-format on

    return tc->Allocate(size);
}

void am_free(void* ptr) {
    // clang-format off
    if (!ptr) AM_UNLIKELY {
        return;
    }

    auto* span = PageMap::GetSpan(ptr);
    if (!span) AM_UNLIKELY {
        return;
    }

    auto size = span->obj_size;
    auto* tc = pTLSThreadCache;
    if (size == 0 || tc == nullptr) AM_UNLIKELY {
        am_free_slow_path(ptr, span, size);
        return;
    }
    // clang-format on

    tc->Deallocate(ptr, size);
}

}// namespace aethermind