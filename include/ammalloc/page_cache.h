//
// Created by richard on 2/7/26.
//

#ifndef AETHERMIND_MALLOC_PAGE_CACHE_H
#define AETHERMIND_MALLOC_PAGE_CACHE_H

#include "ammalloc/page_allocator.h"
#include "ammalloc/span.h"

#include <mutex>

namespace aethermind {

/**
 * @brief Node structure for the Radix Tree (PageMap).
 *
 * Maps Page IDs (keys) to Span pointers (values).
 *
 * @note **Alignment**: `alignas(PAGE_SIZE)` forces the structure to be 4KB aligned.
 * 1. Ensures that one node occupies exactly one physical OS page (assuming 4KB pages).
 * 2. Prevents False Sharing in multi-thread environments.
 * 3. Optimizes interaction with system allocators (like mmap).
 */
struct alignas(SystemConfig::PAGE_SIZE) RadixNode {
    /**
     * @brief Array of pointers to child nodes or Spans.
     *
     * - Size is typically 512 for 64-bit systems (9 bits stride).
     * - In leaf nodes, these point to `Span` objects.
     * - In internal nodes, these point to the next level `RadixNode`.
     */
    std::array<std::atomic<void*>, PageConfig::RADIX_NODE_SIZE> children;

    RadixNode() {
        for (auto& child: children) {
            child.store(nullptr, std::memory_order_relaxed);
        }
    }
};

class PageMap {
public:
    /**
     * @brief Lookup the Span associated with a specific memory address.
     *
     * This function is lock-free and extremely hot in the deallocation path.
     * It relies on the memory barriers established by SetSpan to ensure data visibility.
     *
     * @param page_id The page id being freed or looked up.
     * @return Span* Pointer to the managing Span, or nullptr if not found.
     */
    static Span* GetSpan(size_t page_id);

    static Span* GetSpan(void* ptr) {
        const auto addr = reinterpret_cast<uintptr_t>(ptr);
        const size_t page_id = addr >> SystemConfig::PAGE_SHIFT;
        return GetSpan(page_id);
    }

    /**
    * @brief Register a Span into the PageMap.
    *
    * Associates all page IDs covered by the span with the span pointer.
    * This operation holds a lock to protect the tree structure during growth.
    *
    * @param span The Span to register. Must have valid start_page_idx and page_num.
    */
    static void SetSpan(Span* span);

private:
    // Atomic root pointer for double-checked locking / lazy initialization.
    inline static std::atomic<RadixNode*> root_ = nullptr;
    // Mutex protects tree growth (new node allocation).
    inline static std::mutex mutex_;
    // radix node pool
    inline static ObjectPool<RadixNode> radix_node_pool_{};
};

#ifdef AMMALLOC_TEST
#define PAGE_CACHE_FRIENDS_TEST \
    friend class PageCacheTest;
#else
#define PAGE_CACHE_FRIENDS_TEST
#endif


/**
 * @brief Global singleton managing page-level memory allocation and deallocation.
 *
 * The PageCache is the central repository for Spans (contiguous memory pages).
 * It sits above the OS memory allocator (PageAllocator) and below the CentralCache.
 *
 * Key Responsibilities:
 * 1. **Distribution**: Slices large spans into smaller ones for CentralCache.
 * 2. **Coalescing**: Merges adjacent free spans returned by CentralCache to reduce external fragmentation.
 * 3. **System Interaction**: Requests large memory blocks from the OS when the cache is empty.
 */
class PageCache {
public:
    /**
    * @brief Retrieves the singleton instance of PageCache.
    */
    static PageCache& GetInstance() {
        static PageCache instance;
        return instance;
    }

    // Disable copy and assignment to enforce singleton pattern.
    PageCache(const PageCache&) = delete;
    PageCache& operator=(const PageCache&) = delete;

    /**
     * @brief Allocates a Span with at least `page_num` pages.
     *
     * Thread-safe wrapper that acquires the global lock.
     *
     * @param page_num Number of pages requested.
     * @param obj_size Size of the objects this Span will manage (metadata for CentralCache).
     * @return Pointer to the allocated Span.
     */
    Span* AllocSpan(size_t page_num, size_t obj_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        return AllocSpanLocked(page_num, obj_size);
    }

    /**
     * @brief Returns a Span to the PageCache and attempts to merge it with neighbors.
     *
     * This function performs **Physical Coalescing**:
     * 1. Checks left and right neighbors using the PageMap.
     * 2. If neighbors are free and the total size is within limits, merges them.
     * 3. Inserts the resulting (potentially larger) Span back into the free list.
     *
     * @param span The Span to be released.
     */
    void ReleaseSpan(Span* span) noexcept;

    AM_NODISCARD std::mutex& GetMutex() noexcept {
        return mutex_;
    }

private:
    /// Global lock protecting the span_lists_ structure.
    std::mutex mutex_;
    /// Array of free lists. Index `i` holds Spans of size `i` pages.
    /// Range: [0, MAX_PAGE_NUM], supporting spans up to 128 pages.
    std::array<SpanList, PageConfig::MAX_PAGE_NUM + 1> span_lists_{};
    // object pool for span
    ObjectPool<Span> span_pool_{};

    PageCache() = default;

    /**
     * @brief Internal core logic for allocation (assumes lock is held).
     * Uses a loop to handle system refill and splitting.
     */
    Span* AllocSpanLocked(size_t page_num, size_t obj_size);

    PAGE_CACHE_FRIENDS_TEST;
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_PAGE_CACHE_H
