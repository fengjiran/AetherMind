//
// Created by richard on 2/17/26.
//

#ifndef AETHERMIND_MALLOC_CENTRAL_CACHE_H
#define AETHERMIND_MALLOC_CENTRAL_CACHE_H

#include "ammalloc/page_cache.h"
#include "ammalloc/size_class.h"
#include "ammalloc/span.h"

#include <atomic>
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
    size_t size_;
    size_t max_size_;
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
    size_t FetchRange(FreeList& block_list, size_t batch_num, size_t size) {
        auto idx = SizeClass::Index(size);
        auto& span_list = span_lists_[idx];

        // Apply Bucket Lock (Fine-grained locking).
        std::unique_lock<std::mutex> lock(span_list.GetMutex());

        size_t i = 0;
        void* batch_head = nullptr;
        void* batch_tail = nullptr;
        // Try to fulfill the batch request
        while (i < batch_num) {
            // 1. Refill Logic:
            // If list is empty OR the current head span is fully allocated, get a new span.
            // Note: use_count check is a fast-path hint; AllocObject is the authority.
            if (span_list.empty() ||
                span_list.begin()->use_count.load(std::memory_order_relaxed) >= span_list.begin()->capacity) {
                // GetOneSpan releases the bucket lock internally to avoid deadlock with PageCache.
                if (!GetOneSpan(span_list, size)) {
                    break;
                }
            }

            // 2. Allocation Loop:
            // Take the first span (LRU strategy: valid spans are at front, full ones at back).
            auto* span = span_list.begin();
            while (i < batch_num) {
                void* obj = span->AllocObject();
                if (!obj) {
                    // Current span is full. Move it to the end of the list.
                    // This ensures subsequent allocations check other spans first.
                    span_list.erase(span);
                    span_list.push_back(span);
                    break;// Break inner loop to check the next span or refill.
                }

                // 3. Link objects into a temporary list (LIFO / Head Insert)
                auto* node = static_cast<FreeBlock*>(obj);
                if (batch_head == nullptr) {
                    batch_tail = obj;// First node allocated is the tail of the batch.
                }

                node->next = static_cast<FreeBlock*>(batch_head);
                batch_head = node;
                ++i;
            }
        }

        lock.unlock();
        // 4. Batch Push: Move the collected objects to ThreadCache's FreeList.
        if (i > 0) {
            block_list.push_range(batch_head, batch_tail, i);
        }
        return i;
    }

    /**
    * @brief returns a batch of objects from ThreadCache to CentralCache.
    *
    * Iterates through the list, finds the owning Span for each object via PageMap,
    * and releases the object. May trigger Span release to PageCache.
    *
    * @param start Head of the linked list of objects to release.
    * @param size Size of the objects (must match the bucket).
    */
    void ReleaseListToSpans(void* start, size_t size) {
        auto idx = SizeClass::Index(size);
        auto& span_list = span_lists_[idx];

        std::unique_lock<std::mutex> lock(span_list.GetMutex());
        while (start) {
            void* next = static_cast<FreeBlock*>(start)->next;
            // 1. Identify the Span owning this object.
            auto* span = PageMap::GetSpan(start);
            AM_DCHECK(span != nullptr);
            AM_DCHECK(span->obj_size == size);
            // 2. Return object to Span.
            span->FreeObject(start);

            // 3. Heuristic: If a full span becomes non-full, move it to the front.
            // This allows FetchRange to immediately find this available slot.
            if (span->use_count.load(std::memory_order_relaxed) == span->capacity - 1) {
                span_list.erase(span);
                span_list.push_front(span);
            }

            // 4. Release to PageCache:
            // If the span becomes completely empty, return it to PageCache for coalescing.
            if (span->use_count.load(std::memory_order_relaxed) == 0) {
                span_list.erase(span);
                // Cleanup metadata pointers before returning.
                span->bitmap = nullptr;
                span->data_base_ptr = nullptr;
                // CRITICAL: Unlock bucket lock before calling PageCache to avoid deadlocks.
                // Lock Order: PageCache_Lock > Bucket_Lock (if held together).
                // Here we break the hold.
                lock.unlock();
                PageCache::GetInstance().ReleaseSpan(span);
                // Re-acquire lock to continue processing the list.
                lock.lock();
            }

            start = next;
        }
    }

private:
    CentralCache() = default;

    /**
     * @brief Refills the SpanList by requesting a new Span from PageCache.
     * @warning Must be called with the bucket lock HELD. Will temporarily release it.
     */
    static Span* GetOneSpan(SpanList& list, size_t size) {
        // 1. Unlock bucket lock to perform expensive PageCache operation.
        list.GetMutex().unlock();

        // 2. Calculate optimal page count and request from PageCache.
        auto page_num = SizeClass::GetMovePageNum(size);
        auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
        AM_DCHECK(span != nullptr);
        // 3. Initialize the new Span (Slice it into objects).
        // This accesses new memory, safe to do without lock (thread-local at this point).
        span->Init(size);// NOLINT
        // 4. Re-lock and push to the list.
        list.GetMutex().lock();
        list.push_front(span);
        return span;
    }

    constexpr static size_t kNumSizeClasses = SizeClass::Index(SizeConfig::MAX_TC_SIZE) + 1;
    std::array<SpanList, kNumSizeClasses> span_lists_{};
};


}// namespace aethermind

#endif//AETHERMIND_MALLOC_CENTRAL_CACHE_H
