/// Page-level memory cache and radix-tree page map.
///
/// PageCache manages Spans (contiguous page ranges) and serves as the backend
/// for CentralCache. It handles span splitting, coalescing, and OS allocation.
/// PageMap provides a lock-free read path for span lookup via a 4-level radix tree.
///
/// Architecture:
/// - PageCache sits above PageAllocator (OS interaction) and below CentralCache.
/// - Uses a global mutex for all mutable operations.
/// - Span metadata is pooled to avoid recursive allocation.
///
/// Thread-safety:
/// - PageCache: All public methods acquire the global mutex internally.
/// - PageMap::GetSpan is lock-free; SetSpan/ClearRange require PageCache lock.
///
/// Dependencies: page_allocator.h, span.h
#ifndef AETHERMIND_MALLOC_PAGE_CACHE_H
#define AETHERMIND_MALLOC_PAGE_CACHE_H

#include "ammalloc/page_allocator.h"
#include "ammalloc/span.h"
#include "macros.h"

#include <mutex>

namespace aethermind {

inline uint64_t GetCurrentTimeMs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

/// Root node of the radix tree page map.
/// Aligned to page size to prevent false sharing.
struct alignas(SystemConfig::PAGE_SIZE) RadixRootNode {
    std::array<std::atomic<void*>, PageConfig::RADIX_ROOT_SIZE> children;

    RadixRootNode() {
        for (auto& child : children) {
            child.store(nullptr, std::memory_order_relaxed);
        }
    }
};

/// Internal/leaf node of the radix tree page map.
/// Leaf nodes store Span pointers; internal nodes store child node pointers.
/// Aligned to page size to prevent false sharing.
struct alignas(SystemConfig::PAGE_SIZE) RadixNode {
    std::array<std::atomic<void*>, PageConfig::RADIX_NODE_SIZE> children;

    RadixNode() {
        for (auto& child : children) {
            child.store(nullptr, std::memory_order_relaxed);
        }
    }
};

/// Maps page IDs to managing Spans via a radix tree.
///
/// Thread-safety:
/// - GetSpan is lock-free (hot path in deallocation).
/// - SetSpan, ClearRange, and Reset require PageCache::mutex_ held.
class PageMap {
public:
    /// Lock-free lookup of the Span managing a page.
    ///
    /// @warning Hot path; relies on release semantics in SetSpan.
    static Span* GetSpan(size_t page_id);

    static Span* GetSpan(void* ptr) {
        return GetSpan(reinterpret_cast<uintptr_t>(ptr) >> SystemConfig::PAGE_SHIFT);
    }

    /// Registers all pages of a span in the radix tree.
    ///
    /// @pre PageCache::mutex_ held.
    static void SetSpan(Span* span);

    /// Clears mappings for a page range.
    ///
    /// @pre PageCache::mutex_ held.
    static void ClearRange(size_t start_page_id, size_t page_num);

    /// Resets the entire page map.
    ///
    /// @pre PageCache::mutex_ held.
    static void Reset();

private:
    // Published with release semantics once initialized.
    inline static std::atomic<RadixRootNode*> root_ = nullptr;
    inline static ObjectPool<RadixRootNode> radix_root_pool_{};
    inline static ObjectPool<RadixNode> radix_node_pool_{};
};

#ifdef AMMALLOC_TEST
#define PAGE_CACHE_FRIENDS_TEST \
    friend class PageCacheTest;
#else
#define PAGE_CACHE_FRIENDS_TEST
#endif


/// Global singleton managing page-level memory allocation.
///
/// PageCache manages Spans (contiguous page ranges) as the backend for
/// CentralCache. It handles span splitting, coalescing, and OS allocation.
/// Uses a leaky singleton pattern to avoid destruction order issues.
///
/// Thread-safety: All public methods acquire the global mutex internally.
class PageCache {
public:
    /// Returns the singleton instance.
    static PageCache& GetInstance() {
        // Uses placement new in static storage to avoid recursive allocation.
        alignas(alignof(PageCache)) static char storage[sizeof(PageCache)];
        static auto* instance = new (storage) PageCache();
        return *instance;
    }

    PageCache(const PageCache&) = delete;
    PageCache& operator=(const PageCache&) = delete;

    /// Allocates a span with at least `page_num` pages.
    ///
    /// Thread-safe; acquires the global lock internally.
    /// @return The allocated span, or nullptr if system allocation fails.
    Span* AllocSpan(size_t page_num, size_t obj_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        return AllocSpanLocked(page_num, obj_size);
    }

    /// Returns a span to the cache and attempts coalescing with neighbors.
    ///
    /// Thread-safe; acquires the global lock internally.
    void ReleaseSpan(Span* span) noexcept;

    /// Resets the cache to its initial state.
    ///
    /// Used for test isolation. Clears all spans and resets the page map.
    void Reset();

    AM_NODISCARD bool IsBucketEmpty(size_t bucket_idx) const noexcept {
        AM_DCHECK(bucket_idx < span_lists_.size());
        return span_lists_[bucket_idx].empty();
    }

    AM_NODISCARD std::mutex& GetMutex() noexcept {
        return mutex_;
    }

private:
    // Protected by mutex_.
    std::mutex mutex_;
    /// Free lists indexed by span size in pages. Index 0 is unused.
    std::array<SpanList, PageConfig::MAX_PAGE_NUM + 1> span_lists_{};
    ObjectPool<Span> span_pool_{};

    PageCache() = default;
    ~PageCache() = default;

    /// Core allocation logic. Assumes lock is held.
    Span* AllocSpanLocked(size_t page_num, size_t obj_size);

    PAGE_CACHE_FRIENDS_TEST;
    friend class PageHeapScavenger;
};

}// namespace aethermind

#endif// AETHERMIND_MALLOC_PAGE_CACHE_H
