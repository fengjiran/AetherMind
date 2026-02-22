//
// Created by richard on 2/6/26.
//
#include "ammalloc/thread_cache.h"

namespace aethermind {

void ThreadCache::ReleaseAll() {
    for (size_t i = 0; i < SizeClass::kNumSizeClasses; ++i) {
        auto& list = free_lists_[i];
        if (list.empty()) {
            continue;
        }

        const auto size = SizeClass::Size(i);
        void* start = nullptr;
        const auto cnt = list.size();
        for (size_t j = 0; j < cnt; ++j) {
            void* ptr = list.pop();
            static_cast<FreeBlock*>(ptr)->next = static_cast<FreeBlock*>(start);
            start = ptr;
        }
        CentralCache::GetInstance().ReleaseListToSpans(start, size);
    }
}

void* ThreadCache::FetchFromCentralCache(FreeList& list, size_t size) {
    const auto batch_num = SizeClass::CalculateBatchSize(size);
    auto fetch_num = list.max_size();
    if (fetch_num > batch_num) {
        fetch_num = batch_num;
    }

    // Fetch from CentralCache (This involves locking in CentralCache)
    // 'list' is modified in-place by FetchRange.
    if (CentralCache::GetInstance().FetchRange(list, fetch_num, size) == 0) {
        return nullptr;// Out of memory
    }

    // Dynamic Limit Strategy (Slow Start):
    if (list.max_size() < batch_num * 2) {
        list.set_max_size(list.max_size() + 1);
    }
    return list.pop();
}

void ThreadCache::ReleaseTooLongList(FreeList& list, size_t size) {
    // Strategy: When full, release 'batch_num' objects back to CentralCache.
    // This keeps 'batch_num' objects in ThreadCache (if limit is 2*batch),
    // or empties it if limit == batch.

    // Use the same batch calculation for releasing.
    // We pop 'batch_num' items from the list and link them together.
    auto batch_num = SizeClass::CalculateBatchSize(size);
    if (batch_num == 0 || list.empty()) {
        return;
    }

    void* start = nullptr;
    // Construct a linked list of objects to return
    // We assume FreeList::pop() returns the raw pointer.
    // We use the object's memory to store the 'next' pointer (Embedded List).
    for (size_t i = 0; i < batch_num; ++i) {
        void* ptr = list.pop();
        // Link node: ptr->next = start; start = ptr;
        static_cast<FreeBlock*>(ptr)->next = static_cast<FreeBlock*>(start);
        start = ptr;
    }

    // Send the list to CentralCache
    CentralCache::GetInstance().ReleaseListToSpans(start, size);
}

void ThreadCache::DeallocateSlowPath(FreeList& list, size_t size) {
    const auto batch_num = SizeClass::CalculateBatchSize(size);
    const auto limit = batch_num * 2;
    if (list.max_size() < limit) {
        list.set_max_size(list.max_size() + 1);
    } else {
        void* start = nullptr;
        // Construct a linked list of objects to return
        // We assume FreeList::pop() returns the raw pointer.
        // We use the object's memory to store the 'next' pointer (Embedded List).
        for (size_t i = 0; i < batch_num; ++i) {
            void* ptr = list.pop();
            // Link node: ptr->next = start; start = ptr;
            static_cast<FreeBlock*>(ptr)->next = static_cast<FreeBlock*>(start);
            start = ptr;
        }

        // Send the list to CentralCache
        CentralCache::GetInstance().ReleaseListToSpans(start, size);
    }
}

}// namespace aethermind