//
// Created by richard on 2/17/26.
//

#include "ammalloc/central_cache.h"
#include "ammalloc/page_cache.h"
#include "ammalloc/spin_lock.h"

#include <cstddef>
#include <vector>

namespace aethermind {

size_t CentralCache::FetchRange(FreeList& block_list, size_t batch_num, size_t size) {
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
        auto& bucket = buckets_[i];
        // ===================================================================
        // 1. 清空 TransferCache (Fast Path 缓存)
        // ===================================================================
        // 使用栈上数组暂存指针，避免使用 std::vector 触发 malloc
        // 最大的 tc_capacity 是 512 * 2 = 1024，1024 个指针占用 8KB 栈空间，非常安全
        void* local_transfer_cache[1024];
        size_t local_transfer_cache_count = 0;
        bucket.transfer_cache_lock.lock();
        local_transfer_cache_count = bucket.transfer_cache_count;
        for (size_t j = 0; j < local_transfer_cache_count; ++j) {
            local_transfer_cache[j] = bucket.transfer_cache[j];
        }
        bucket.transfer_cache_count = 0;
        bucket.transfer_cache_lock.unlock();

        // ===================================================================
        // 2. 将对象还给 Span，并清空 SpanList
        // ===================================================================
        Span* span_list_head = nullptr;
        {
            std::lock_guard<std::mutex> lock(bucket.span_list_lock);
            // A. 恢复 Span 的使用计数 (将刚才从 TransferCache 拿出的对象还回去)
            for (size_t j = 0; j < local_transfer_cache_count; ++j) {
                void* obj = local_transfer_cache[j];
                auto* span = PageMap::GetSpan(obj);
                if (span) {
                    span->FreeObject(obj);
                }
            }

            // B. 掏空 SpanList
            while (!bucket.span_list.empty()) {
                auto* span = bucket.span_list.pop_front();
                // 清理元数据
                span->bitmap = nullptr;
                span->data_base_ptr = nullptr;
                // 利用 Span 自身的 next 指针串成临时链表，避免使用 std::vector
                span->next = span_list_head;
                span_list_head = span;
            }
        }// 解锁 span_lock

        // ===================================================================
        // 3. 将所有 Span 归还给 PageCache
        // ===================================================================
        while (span_list_head) {
            auto* next_span = span_list_head->next;
            PageCache::GetInstance().ReleaseSpan(span_list_head);
            span_list_head = next_span;
        }
    }

    // ===================================================================
    // 4. 彻底释放 TransferCache 占用的底层物理内存
    // ===================================================================
    // buckets_[0].tc_objects 指向的是 InitTransferCaches 时申请的连续大块内存的起始位置
    if (buckets_[0].transfer_cache) {
        // 重新计算当时申请的页数
        size_t total_ptrs = 0;
        for (size_t i = 0; i < kNumSizeClasses; ++i) {
            size_t batch_num = SizeClass::CalculateBatchSize(SizeClass::Size(i));
            total_ptrs += 2 * batch_num;
        }
        size_t total_bytes = total_ptrs * sizeof(void*);
        size_t page_num = (total_bytes + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
        PageAllocator::SystemFree(buckets_[0].transfer_cache, page_num);

        // 重置所有桶的指针，防止 UAF
        for (size_t i = 0; i < kNumSizeClasses; ++i) {
            auto& bucket = buckets_[i];
            bucket.transfer_cache = nullptr;
            bucket.transfer_cache_capacity = 0;
            bucket.transfer_cache_count = 0;
        }
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
