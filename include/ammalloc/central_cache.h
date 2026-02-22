//
// Created by richard on 2/17/26.
//

#ifndef AETHERMIND_MALLOC_CENTRAL_CACHE_H
#define AETHERMIND_MALLOC_CENTRAL_CACHE_H

#include "ammalloc/size_class.h"
#include "ammalloc/span.h"

#include <cstdint>
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
 * CentralCache acts as a hub that balances memory resources among multiple threads.
 * It divides memory into different "Size Classes" (Buckets), each protected by a separate lock (Bucket Lock).
 *
 * Key Responsibilities:
 * 1. **Distribution**: Fetches large Spans from PageCache, slices them into objects, and serves ThreadCache in batches.
 * 2. **Recycling**: Receives returned objects from ThreadCache and releases Spans back to PageCache when they are completely empty.
 * 3. **Concurrency**: Reduces lock contention using fine-grained bucket locks compared to the single global lock in PageCache.
 */
class CentralCache {
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
    * @brief returns a batch of objects from ThreadCache to CentralCache.
    *
    * Iterates through the list, finds the owning Span for each object via PageMap,
    * and releases the object. May trigger Span release to PageCache.
    *
    * @param start Head of the linked list of objects to release.
    * @param size Size of the objects (must match the bucket).
    */
    void ReleaseListToSpans(void* start, size_t size);

    void Reset() noexcept;

private:
    CentralCache() = default;

    /**
     * @brief Refills the SpanList by requesting a new Span from PageCache.
     * @warning Must be called with the bucket lock HELD. Will temporarily release it.
     */
    static Span* GetOneSpan(SpanList& list, size_t size, std::unique_lock<std::mutex>& lock);

    constexpr static size_t kNumSizeClasses = SizeClass::Index(SizeConfig::MAX_TC_SIZE) + 1;
    std::array<SpanList, kNumSizeClasses> span_lists_{};
};


}// namespace aethermind

#endif//AETHERMIND_MALLOC_CENTRAL_CACHE_H
