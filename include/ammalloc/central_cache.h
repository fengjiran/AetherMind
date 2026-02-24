//
// Created by richard on 2/17/26.
//

#ifndef AETHERMIND_MALLOC_CENTRAL_CACHE_H
#define AETHERMIND_MALLOC_CENTRAL_CACHE_H

#include "ammalloc/size_class.h"
#include "ammalloc/span.h"
#include "ammalloc/spin_lock.h"

#include <cstddef>
#include <mutex>

namespace aethermind {

struct FreeBlock {
    FreeBlock* next;
};

class FreeList {
public:
    constexpr FreeList() noexcept : head_(nullptr), size_(0), max_size_(1) {}

    FreeList(const FreeList&) = delete;
    FreeList& operator=(const FreeList&) = delete;

    AM_NODISCARD bool empty() const noexcept {
        return head_ == nullptr;
    }

    AM_NODISCARD size_t size() const noexcept {
        return size_;
    }

    void clear() noexcept {
        head_ = nullptr;
        size_ = 0;
    }

    void push(void* ptr) noexcept {
        if (!ptr) AM_UNLIKELY return;

        auto* block = static_cast<FreeBlock*>(ptr);
        block->next = head_;
        head_ = block;
        ++size_;
    }

    void push_range(void* begin, void* end, size_t count) noexcept {
        if (!begin || !end || count == 0) {
            return;
        }

        static_cast<FreeBlock*>(end)->next = head_;
        head_ = static_cast<FreeBlock*>(begin);
        size_ += count;
    }

    AM_NODISCARD void* pop() noexcept {
        if (empty()) AM_UNLIKELY return nullptr;

        auto* block = head_;
        if (block->next) AM_LIKELY AM_BUILTIN_PREFETCH(block->next, 0, 3);

        head_ = head_->next;
        --size_;
        return block;
    }

    AM_NODISCARD size_t max_size() const noexcept {
        return max_size_;
    }

    void set_max_size(size_t n) noexcept {
        max_size_ = n;
    }

private:
    FreeBlock* head_;
    uint32_t size_;
    uint32_t max_size_;
};

/**
 * @brief Central resource manager connecting ThreadCache and PageCache.
 *
 * CentralCache acts as a global hub that balances memory resources among multiple threads.
 * It implements a highly optimized two-tier caching strategy per Size Class to minimize
 * lock contention and maximize throughput during batch operations.
 *
 * ### Two-Tier Bucket Architecture:
 * 1. **Transfer Cache (Fast Path)**: A lock-free/spin-locked array of pointers. It acts as a
 *    buffer to quickly absorb released objects and serve allocation requests without touching
 *    complex metadata (like Bitmaps or PageMaps).
 * 2. **Span List (Slow Path)**: A mutex-locked doubly linked list of Spans. It is only accessed
 *    when the Transfer Cache is exhausted or full, handling the actual slicing of Spans and
 *    interacting with the global PageCache.
 */
class CentralCache {
    /**
     * @brief Manages objects of a specific Size Class.
     * @note Aligned to the cache line size (e.g., 64 bytes) to completely eliminate
     *       False Sharing between threads accessing different buckets concurrently.
     */
    struct alignas(SystemConfig::CACHE_LINE_SIZE) Bucket {
        // --- Tier 1: Transfer Cache (Fast Path) ---
        /// Lightweight spinlock protecting the pointer array.
        SpinLock transfer_cache_lock;
        /// Current number of cached object pointers.
        size_t transfer_cache_count{0};
        /// Dynamic capacity, usually configured as 2x the batch size.
        size_t transfer_cache_capacity{0};
        /// Pointer to a dynamically allocated array of object pointers.
        void** transfer_cache{nullptr};

        // --- Tier 2: Span List (Slow Path) ---
        /// Heavyweight mutex protecting the Span list and bitmap operations.
        std::mutex span_list_lock;
        /// Pure, lock-free doubly linked list of Spans.
        SpanList span_list;
    };

public:
    /**
     * @brief Singleton Accessor.
     */
    static CentralCache& GetInstance() {
        static CentralCache instance;
        return instance;
    }

    // Disable copy/move to enforce singleton pattern.
    CentralCache(const CentralCache&) = delete;
    CentralCache& operator=(const CentralCache&) = delete;

    /**
     * @brief Fetches a batch of objects for a specific ThreadCache.
     *
     * This function pulls objects from the non-empty spans in the corresponding bucket.
     * If the bucket is empty or exhausted, it requests a new Span from PageCache.
     *
     * @param block_list Output parameter. The fetched objects are pushed into this FreeList.
     * @param batch_num The desired number of objects to fetch.
     * @param size The size of the object (used to determine the bucket index).
     * @return size_t The actual number of objects fetched (may be less than batch_num).
     */
    size_t FetchRange(FreeList& block_list, size_t batch_num, size_t size);

    /**
     * @brief Fetches a batch of objects to refill a ThreadCache.
     *
     * Prioritizes fetching from the fast TransferCache. If insufficient, falls back
     * to slicing objects from the SpanList.
     *
     * @param block_list Output parameter. The fetched objects are pushed into this FreeList.
     * @param batch_num The desired number of objects to fetch.
     * @param size The size of the object (used to determine the bucket index).
     * @return size_t The actual number of objects fetched (may be less than batch_num if OOM).
     */
    size_t FetchRange1(FreeList& block_list, size_t batch_num, size_t size);

    /**
    * @brief returns a batch of objects from ThreadCache to CentralCache.
    *
    * Iterates through the list, finds the owning Span for each object via PageMap,
    * and releases the object. May trigger Span release to PageCache.
    *
    * @param start Head of the linked list of objects to release.
    * @param size Size of the objects (must match the bucket).
    */
    void ReleaseListToSpans(void* start, size_t size);

    /**
     * @brief Returns a batch of objects from a ThreadCache back to the CentralCache.
     *
     * Prioritizes pushing objects into the fast TransferCache. Any overflow is
     * returned to their respective Spans, potentially triggering a release to PageCache.
     *
     * @param start Head of the linked list of objects to release.
     * @param size Size of the objects (must match the bucket).
     */
    void ReleaseListToSpans1(void* start, size_t size);

    void Reset() noexcept;
    void Reset1() noexcept;

private:
    /**
     * @brief Private constructor. Initializes the dynamic TransferCache arrays.
     */
    CentralCache() {
        InitTransferCache();
    }

    /**
     * @brief Allocates a contiguous block of memory from the OS to back all TransferCaches.
     * @note Bypasses the standard am_malloc to prevent initialization deadlocks.
     */
    void InitTransferCache();

    /**
     * @brief Refills the SpanList by requesting a new Span from PageCache.
     * @warning Must be called with the bucket lock HELD. Will temporarily release it.
     */
    static Span* GetOneSpan(SpanList& list, size_t size, std::unique_lock<std::mutex>& lock);

    /**
     * @brief Refills the SpanList by requesting a new Span from PageCache.
     * @warning Must be called with the `span_lock` HELD. Will temporarily release it
     *          to prevent deadlocks with the global PageCache lock.
     */
    static Span* GetOneSpan(Bucket& bucket, size_t size, std::unique_lock<std::mutex>& lock);

    constexpr static size_t kNumSizeClasses = SizeClass::Index(SizeConfig::MAX_TC_SIZE) + 1;
    std::array<SpanList, kNumSizeClasses> span_lists_{};
    /// Array of Buckets (The Hash Table).
    std::array<Bucket, kNumSizeClasses> buckets_{};
};


}// namespace aethermind

#endif//AETHERMIND_MALLOC_CENTRAL_CACHE_H
