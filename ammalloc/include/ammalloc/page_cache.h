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
        for (auto& child: children) {
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
        for (auto& child: children) {
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
    inline static RadixRootNode root_storage_{};
    inline static ObjectPool<RadixNode> radix_node_pool_{};
};

#ifdef AMMALLOC_TEST
#define PAGE_CACHE_FRIENDS_TEST \
    friend class PageCacheTest;
#else
#define PAGE_CACHE_FRIENDS_TEST
#endif

/// One shard of PageCache free-span state.
///
/// Minimal design:
/// - One mutex per shard
/// - One bucket array per shard
/// - One Span metadata pool per shard
///
/// First-stage rule:
/// - Allocation and release always happen through the owning shard
/// - Coalescing is only allowed within the owner shard
class alignas(SystemConfig::CACHE_LINE_SIZE) PageCacheShard {
public:
    PageCacheShard() = default;
    ~PageCacheShard() = default;

    PageCacheShard(const PageCacheShard&) = delete;
    PageCacheShard& operator=(const PageCacheShard&) = delete;

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

    /// Core allocation logic. Assumes lock is held.
    Span* AllocSpanLocked(size_t page_num);

    /// Core release path. Caller must hold mutex_.
    void ReleaseSpanLocked(Span* span) noexcept;

    /// Reset this shard only. Caller must hold mutex_ and ensure quiescent state.
    void ResetLocked();

    friend class PageCache;
    PAGE_CACHE_FRIENDS_TEST;
    friend class PageHeapScavenger;
};

/// Global PageCache manager.
///
/// External API remains close to the current single-instance design, but
/// internal mutable state is partitioned into shards.
///
/// Current rollout plan:
/// - active_shard_count_ may remain 1 on single-NUMA/dev machines
/// - later it can be increased for logical sharding or NUMA-aware sharding
///
/// Ownership rule:
/// - Every Span has an owner_shard_id
/// - Release always routes back to owner shard
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
    /// Current policy:
    /// - Select a shard
    /// - Lock only that shard
    /// - Allocate from that shard's free lists / pool
    ///
    /// @return The allocated span, or nullptr if system allocation fails.
    Span* AllocSpan(size_t page_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        return AllocSpanLocked(page_num);
    }

    /// Return a span to its owner shard.
    ///
    /// Current policy:
    /// - Owner shard is recorded in Span::owner_shard_id
    /// - Release/coalescing happens only inside owner shard
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
    Span* AllocSpanLocked(size_t page_num);

    PAGE_CACHE_FRIENDS_TEST;
    friend class PageHeapScavenger;
};

}// namespace aethermind

#endif// AETHERMIND_MALLOC_PAGE_CACHE_H
