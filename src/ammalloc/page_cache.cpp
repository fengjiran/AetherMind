//
// Created by richard on 2/7/26.
//

#include "ammalloc/page_cache.h"

namespace aethermind {

Span* PageCache::AllocSpanLocked(size_t page_num, size_t obj_size) {
    while (true) {
        // 1. Oversized Allocation:
        // Requests larger than the max bucket (>128 pages) go directly to the OS.
        // clang-format off
        if (page_num > PageConfig::MAX_PAGE_NUM) AM_UNLIKELY {
            void* ptr = PageAllocator::SystemAlloc(page_num);
            if (!ptr) {
                return nullptr;
            }

            auto* span = span_pool_.New();
            span->start_page_idx = reinterpret_cast<uintptr_t>(ptr) >> SystemConfig::PAGE_SHIFT;
            span->page_num = page_num;
            span->obj_size = obj_size;
            span->is_used = true;

            // Register relationship in Radix Tree.
            PageMap::SetSpan(span);
            return span;
        }
        // clang-format on

        // 2. Exact Match:
        // Check if there is a free span in the bucket corresponding exactly to page_num.
        if (!span_lists_[page_num].empty()) {
            auto* span = span_lists_[page_num].pop_front();
            span->obj_size = obj_size;
            span->is_used = true;
            return span;
        }

        // 3. Splitting (Best Fit / First Fit):
        // Iterate through larger buckets to find a span we can split.
        for (size_t i = page_num + 1; i <= PageConfig::MAX_PAGE_NUM; ++i) {
            if (span_lists_[i].empty()) {
                continue;
            }

            // Found a larger span.
            auto* big_span = span_lists_[i].pop_front();
            // Create a new span for the requested `page_num` (Head Split).
            auto* small_span = span_pool_.New();
            small_span->start_page_idx = big_span->start_page_idx;
            small_span->page_num = page_num;
            small_span->obj_size = obj_size;
            small_span->is_used = true;

            // Adjust the remaining part of the big span (Tail).
            big_span->start_page_idx += page_num;
            big_span->page_num -= page_num;
            big_span->is_used = false;
            // Return the remainder to the appropriate free list.
            span_lists_[big_span->page_num].push_front(big_span);

            // Register both parts in the PageMap.
            PageMap::SetSpan(small_span);
            PageMap::SetSpan(big_span);
            return small_span;
        }

        // 4. System Refill:
        // If no suitable spans exist in cache, allocate a large block (128 pages) from OS.
        // We request the MAX_PAGE_NUM to maximize cache efficiency.
        size_t alloc_page_nums = PageConfig::MAX_PAGE_NUM;
        void* ptr = PageAllocator::SystemAlloc(alloc_page_nums);
        auto* span = span_pool_.New();
        span->start_page_idx = reinterpret_cast<uintptr_t>(ptr) >> SystemConfig::PAGE_SHIFT;
        span->page_num = alloc_page_nums;
        span->is_used = false;
        // Insert the new large span into the last bucket.
        span_lists_[alloc_page_nums].push_front(span);
        PageMap::SetSpan(span);
        // Continue the loop:
        // The next iteration will jump to step 3 (Splitting), finding the
        // 128-page span we just added, splitting it, and returning the result.
    }
}

void PageCache::ReleaseSpan(Span* span) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Direct Return: If the Span is larger than the cache can manage (>128 pages),
    // return it directly to the OS (PageAllocator).
    // clang-format off
    if (span->page_num > PageConfig::MAX_PAGE_NUM) AM_UNLIKELY {
        auto* ptr = span->GetStartAddr();
        PageAllocator::SystemFree(ptr, span->page_num);
        span_pool_.Delete(span);
        return;
    }
    // clang-format on

    // 2. Merge Left: Check the previous page ID.
    while (true) {
        size_t left_id = span->start_page_idx - 1;
        // Retrieve the Span managing the left page from the global PageMap.
        auto* left_span = PageMap::GetSpan(left_id);
        // Stop merging if:
        // - Left page doesn't exist (not managed by us).
        // - Left span is currently in use (in CentralCache).
        // - Merged size would exceed the maximum bucket size.
        if (!left_span || left_span->is_used ||
            span->page_num + left_span->page_num > PageConfig::MAX_PAGE_NUM) {
            break;
        }

        // Perform merge: Remove left_span from its list, absorb it into 'span'.
        span_lists_[left_span->page_num].erase(left_span);
        span->start_page_idx = left_span->start_page_idx;// Update start to left
        span->page_num += left_span->page_num;           // Increase size
        span_pool_.Delete(left_span);                    // Destroy metadata
    }

    // 3. Merge Right: Check the page ID immediately following this span.
    while (true) {
        size_t right_id = span->start_page_idx + span->page_num;
        auto* right_span = PageMap::GetSpan(right_id);
        // Similar stop conditions as Merge Left.
        if (!right_span || right_span->is_used ||
            span->page_num + right_span->page_num > PageConfig::MAX_PAGE_NUM) {
            break;
        }

        // Perform merge: Remove right_span, absorb it.
        span_lists_[right_span->page_num].erase(right_span);
        span->page_num += right_span->page_num;// Start index stays same, size increases
        span_pool_.Delete(right_span);
    }

    // 4. Insert back: Mark as unused and push to the appropriate bucket.
    span->is_used = false;
    span->obj_size = 0;
    span_lists_[span->page_num].push_front(span);
    // Update PageMap: Map ALL pages in this coalesced span to the span pointer.
    // This ensures subsequent merge operations can find this span via any of its pages.
    PageMap::SetSpan(span);
}

}// namespace aethermind
