//
// Created by richard on 2/17/26.
//

#include "ammalloc/central_cache.h"
#include "ammalloc/page_cache.h"
#include "ammalloc/spin_lock.h"

#include <vector>

namespace aethermind {

size_t CentralCache::FetchRange(FreeList& block_list, size_t batch_num, size_t size) {
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
            span_list.begin()->use_count >= span_list.begin()->capacity) {
            // GetOneSpan releases the bucket lock internally to avoid deadlock with PageCache.
            if (!GetOneSpan(span_list, size, lock)) {
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

size_t CentralCache::FetchRange1(FreeList& block_list, size_t batch_num, size_t size) {
    auto idx = SizeClass::Index(size);
    auto& bucket = buckets_[idx];

    size_t fetched = 0;
    void* head = nullptr;
    void* tail = nullptr;

    // 1. Fast Path: Extract from TransferCache (SpinLock)
    bucket.transfer_cache_lock.lock();
    size_t grab_count = std::min(batch_num, bucket.transfer_cache_count);
    for (size_t i = 0; i < grab_count; ++i) {
        void* obj = bucket.transfer_cache[--bucket.transfer_cache_count];
        auto* node = static_cast<FreeBlock*>(obj);
        // Build a temporary linked list (LIFO / Head Insert)
        if (!head) {
            tail = obj;
        }
        node->next = static_cast<FreeBlock*>(head);
        head = node;
    }
    fetched = grab_count;
    bucket.transfer_cache_lock.unlock();

    // 2. Slow Path: Extract from SpanList (Mutex + Bitmap Operations)
    if (fetched < batch_num) {
        std::unique_lock<std::mutex> lock(bucket.span_list_lock);
        while (fetched < batch_num) {
            // Refill SpanList if empty or current head span is fully utilized.
            if (bucket.span_list.empty() ||
                bucket.span_list.begin()->use_count >= bucket.span_list.begin()->capacity) {
                if (!GetOneSpan(bucket, size, lock)) {
                    break;// OOM
                }
            }

            auto* span = bucket.span_list.begin();
            while (fetched < batch_num) {
                void* obj = span->AllocObject();// Heavy bitmap scanning
                if (!obj) {
                    // Span is full. Move it to the back (LRU strategy).
                    bucket.span_list.erase(span);
                    bucket.span_list.push_back(span);
                    break;
                }

                auto* node = static_cast<FreeBlock*>(obj);
                if (!head) {
                    tail = obj;
                }
                node->next = static_cast<FreeBlock*>(head);
                head = node;
                ++fetched;
            }
        }
    }

    // 3. Deliver the constructed list to ThreadCache
    if (fetched > 0) {
        block_list.push_range(head, tail, fetched);
    }
    return fetched;
}

void CentralCache::ReleaseListToSpans(void* start, size_t size) {
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
        if (span->use_count == span->capacity - 1) {
            span_list.erase(span);
            span_list.push_front(span);
        }

        // 4. Release to PageCache:
        // If the span becomes completely empty, return it to PageCache for coalescing.
        if (span->use_count == 0) {
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

void CentralCache::ReleaseListToSpans1(void* start, size_t size) {
    auto idx = SizeClass::Index(size);
    auto& bucket = buckets_[idx];

    void* cur = start;
    // 1. Fast Path: Push into TransferCache (SpinLock)
    bucket.transfer_cache_lock.lock();
    while (cur && bucket.transfer_cache_count < bucket.transfer_cache_capacity) {
        void* next = static_cast<FreeBlock*>(cur)->next;
        bucket.transfer_cache[bucket.transfer_cache_count++] = cur;
        cur = next;
    }
    bucket.transfer_cache_lock.unlock();

    // 2. Slow Path: Return to SpanList (Mutex + PageMap Lookups)
    if (cur) {
        std::unique_lock<std::mutex> lock(bucket.span_list_lock);
        while (cur) {
            void* next = static_cast<FreeBlock*>(cur)->next;
            // Heavy operations: Radix Tree lookup and Bitmap modification
            auto* span = PageMap::GetSpan(cur);
            AM_DCHECK(span != nullptr);
            AM_DCHECK(span->obj_size == size);
            span->FreeObject(cur);

            // If a full span becomes non-full, move it to the front.
            // This allows FetchRange to immediately find this available slot.
            if (span->use_count == span->capacity - 1) {
                bucket.span_list.erase(span);
                bucket.span_list.push_front(span);
            }

            // If the span becomes completely empty, return it to PageCache for coalescing.
            if (span->use_count == 0) {
                bucket.span_list.erase(span);
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
            cur = next;
        }
    }
}

void CentralCache::Reset() noexcept {
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        auto& span_list = span_lists_[i];
        std::vector<Span*> spans_to_release;
        {
            std::lock_guard<std::mutex> lock(span_list.GetMutex());
            while (!span_list.empty()) {
                auto* span = span_list.pop_front();
                span->bitmap = nullptr;
                span->data_base_ptr = nullptr;
                spans_to_release.push_back(span);
            }
        }
        // Release spans to PageCache.
        for (auto* span: spans_to_release) {
            PageCache::GetInstance().ReleaseSpan(span);
        }
    }
}

void CentralCache::Reset1() noexcept {
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        auto& bucket = buckets_[i];
        // 1. Clear TransferCache(Fast Path)

    }
}

void CentralCache::InitTransferCache() {
    // 1. Calculate the total number of pointers needed across all buckets.
    size_t total_ptrs = 0;
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        size_t batch_num = SizeClass::CalculateBatchSize(SizeClass::Size(i));
        // Strategy: TransferCache capacity is 2x the batch size to provide a high-water mark buffer.
        total_ptrs += 2 * batch_num;
    }

    // 2. Request a single, large contiguous block from the system allocator.
    // This avoids calling am_malloc during initialization (preventing infinite recursion).
    size_t total_bytes = total_ptrs * sizeof(void*);
    size_t page_num = (total_bytes + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
    void* p = PageAllocator::SystemAlloc(page_num);
    if (!p) {
        spdlog::critical("CentralCache failed to allocate memory for TransferCaches!");
        std::abort();
    }

    // 3. Partition the allocated memory among the buckets.
    auto** cur_ptr = static_cast<void**>(p);
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        size_t batch_num = SizeClass::CalculateBatchSize(SizeClass::Size(i));
        buckets_[i].transfer_cache_capacity = batch_num * 2;
        buckets_[i].transfer_cache = cur_ptr;
        cur_ptr += buckets_[i].transfer_cache_capacity;
    }
}

Span* CentralCache::GetOneSpan(SpanList& list, size_t size, std::unique_lock<std::mutex>& lock) {
    lock.unlock();

    auto page_num = SizeClass::GetMovePageNum(size);
    auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
    AM_DCHECK(span != nullptr);
    span->Init(size);
    lock.lock();
    list.push_front(span);
    return span;
}

Span* CentralCache::GetOneSpan(Bucket& bucket, size_t size, std::unique_lock<std::mutex>& lock) {
    lock.unlock();
    auto page_num = SizeClass::GetMovePageNum(size);
    auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
    AM_DCHECK(span != nullptr);
    span->Init(size);
    lock.lock();
    bucket.span_list.push_front(span);
    return span;
}

}// namespace aethermind
