//
// Created by richard on 2/17/26.
//

#include "ammalloc/central_cache.h"
#include "ammalloc/page_cache.h"

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

}// namespace aethermind
