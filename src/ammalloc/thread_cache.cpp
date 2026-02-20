//
// Created by richard on 2/6/26.
//
#include "ammalloc/thread_cache.h"

namespace aethermind {

void* ThreadCache::Allocate(size_t size) noexcept {
    AM_DCHECK(size <= SizeConfig::MAX_TC_SIZE);
    size_t idx = SizeClass::Index(size);
    auto& list = free_lists_[idx];
    // 1. Fast Path: Pop from local free list (Lock-Free)
    // clang-format off
    if (!list.empty()) AM_LIKELY {
        return list.pop();
    }
    // clang-format on

    // 2. Slow Path: Fetch from CentralCache
    // Note: We must pass the aligned size to CentralCache/PageCache logic
    return FetchFromCentralCache(list, SizeClass::RoundUp(size));
}

void ThreadCache::Deallocate(void* ptr, size_t size) {
    AM_DCHECK(ptr != nullptr);
    AM_DCHECK(size <= SizeConfig::MAX_TC_SIZE);

    size_t idx = SizeClass::Index(size);
    auto& list = free_lists_[idx];
    // 1. Fast Path: Push to local free list (Lock-Free)
    list.push(ptr);

    // 2. Slow Path: Return memory if cache is too large (Scavenging)
    // If the list length exceeds the limit, return a batch to CentralCache.
    if (list.size() >= list.max_size()) {
        const auto limit = SizeClass::CalculateBatchSize(size) * 2;
        if (list.max_size() < limit) {
            list.set_max_size(list.max_size() + 1);
        } else {
            ReleaseTooLongList(list, size);
        }
    }
}

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
    const auto limit = SizeClass::CalculateBatchSize(size);
    auto batch_num = list.max_size();
    if (batch_num > limit) {
        batch_num = limit;
    }

    // Fetch from CentralCache (This involves locking in CentralCache)
    // 'list' is modified in-place by FetchRange.
    auto actual_num = CentralCache::GetInstance().FetchRange(list, batch_num, size);
    if (actual_num == 0) {
        return nullptr;// Out of memory
    }

    // Dynamic Limit Strategy (Slow Start):
    if (list.max_size() < limit * 2) {
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

}// namespace aethermind