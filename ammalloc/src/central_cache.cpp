//
// Created by richard on 2/17/26.
//
#include "ammalloc/central_cache.h"
#include "ammalloc/page_cache.h"
#include "ammalloc/spin_lock.h"

namespace aethermind {

size_t CentralCache::FetchRange(FreeList& block_list, size_t batch_num, size_t obj_size) {
    AM_DCHECK(batch_num <= SizeClass::kMaxBatchSize);
    auto idx = SizeClass::Index(obj_size);
    auto& bucket = buckets_[idx];

    void* local_ptrs[SizeClass::kMaxBatchSize];
    size_t fetched = 0;

    // 1. Fast Path: Extract from TransferCache (SpinLock)
    bucket.transfer_cache_lock.lock();
    size_t grab_count = std::min(batch_num, bucket.transfer_cache_count);
    for (size_t i = 0; i < grab_count; ++i) {
        local_ptrs[i] = bucket.transfer_cache[--bucket.transfer_cache_count];
    }
    bucket.transfer_cache_lock.unlock();

    fetched = grab_count;
    void* head = nullptr;
    void* tail = nullptr;
    for (size_t i = fetched; i > 0; --i) {
        void* obj = local_ptrs[i - 1];
        auto* node = static_cast<FreeBlock*>(obj);
        // Build a temporary linked list (LIFO / Head Insert)
        if (!head) {
            tail = obj;
        }
        node->next = static_cast<FreeBlock*>(head);
        head = node;
    }

    // 2. Slow Path: Extract from SpanList (Mutex + Bitmap Operations) + Prefetching
    if (fetched < batch_num) {
        size_t need_for_thread = batch_num - fetched;
        // 【核心】：预取目标，额外多拿一整个 batch 放进 TransferCache 中备用
        // 这样下个线程来就能直接命中 Fast Path
        // TODO: Future Optimization - Dynamically adjust prefetch_target based on TransferCache's remaining capacity
        // e.g., prefetch_target = std::min(batch_num, bucket.transfer_cache_capacity - bucket.transfer_cache_count);
        size_t prefetch_target = batch_num;
        size_t total_to_extract = need_for_thread + prefetch_target;
        // 用于暂存预取指针的栈数组
        void* prefetch_ptrs[SizeClass::kMaxBatchSize];
        size_t actual_prefetched = 0;
        size_t total_extracted = 0;

        std::unique_lock<std::mutex> lock(bucket.span_list_lock);
        while (total_extracted < total_to_extract) {
            // Refill SpanList if empty or current head span is fully utilized.
            if (bucket.span_list.empty() ||
                bucket.span_list.begin()->use_count >= bucket.span_list.begin()->capacity) {
                if (!GetOneSpan(bucket, obj_size, lock)) {
                    break;// OOM
                }
            }

            auto* span = bucket.span_list.begin();
            while (total_extracted < total_to_extract) {
                void* obj = span->AllocObject();// Heavy bitmap scanning
                if (!obj) {
                    // Span is full. Move it to the back (LRU strategy).
                    bucket.span_list.erase(span);
                    bucket.span_list.push_back(span);
                    break;
                }

                // dispatch
                if (total_extracted < need_for_thread) {
                    // 前半部分：直接交给请求的线程
                    auto* node = static_cast<FreeBlock*>(obj);
                    if (!head) {
                        tail = obj;
                    }
                    node->next = static_cast<FreeBlock*>(head);
                    head = node;
                    ++fetched;
                } else {
                    // 后半部分：暂存起来，准备塞入 TransferCache
                    prefetch_ptrs[actual_prefetched++] = obj;
                }
                ++total_extracted;
            }
        }
        // 【关键】释放重量级 Mutex
        lock.unlock();

        // 3. 将预取的对象推入 TransferCache
        if (actual_prefetched > 0) {
            size_t successfully_pushed = 0;
            bucket.transfer_cache_lock.lock();
            // 只要 TransferCache 还有空间，就塞进去
            while (successfully_pushed < actual_prefetched &&
                   bucket.transfer_cache_count < bucket.transfer_cache_capacity) {
                bucket.transfer_cache[bucket.transfer_cache_count++] = prefetch_ptrs[successfully_pushed++];
            }
            bucket.transfer_cache_lock.unlock();

            // 4. 并发极端情况兜底 (Fallback)
            // 如果在我们拿 Mutex 期间，有别的线程把 TransferCache 还满了
            // 导致我们预取的指针没塞完，需要把多余的安全退回去
            if (successfully_pushed < actual_prefetched) {
                void* leftover_head = nullptr;

                // 串成链表
                for (size_t i = successfully_pushed; i < actual_prefetched; ++i) {
                    auto* node = static_cast<FreeBlock*>(prefetch_ptrs[i]);
                    node->next = static_cast<FreeBlock*>(leftover_head);
                    leftover_head = prefetch_ptrs[i];
                }

                // 复用成熟的还款逻辑，它会处理好一切
                ReleaseListToSpans(leftover_head, obj_size);
            }
        }
    }

    // 5. Deliver the constructed list to ThreadCache
    if (fetched > 0) {
        block_list.push_range(head, tail, fetched);
    }
    return fetched;
}

void CentralCache::ReleaseListToSpans(void* start, size_t obj_size) {
    auto idx = SizeClass::Index(obj_size);
    auto& bucket = buckets_[idx];
    void* cur = start;

    while (cur) {
        void* local_ptrs[SizeClass::kMaxBatchSize];
        size_t local_count = 0;
        while (cur && local_count < SizeClass::kMaxBatchSize) {
            local_ptrs[local_count++] = cur;
            cur = static_cast<FreeBlock*>(cur)->next;
        }

        // 1. Fast Path: Push into TransferCache (SpinLock)
        size_t pushed = 0;
        bucket.transfer_cache_lock.lock();
        while (pushed < local_count && bucket.transfer_cache_count < bucket.transfer_cache_capacity) {
            bucket.transfer_cache[bucket.transfer_cache_count++] = local_ptrs[pushed++];
        }
        bucket.transfer_cache_lock.unlock();

        // 2. Slow Path: Return to SpanList (Mutex + PageMap Lookups)
        if (pushed < local_count) {
            std::unique_lock<std::mutex> lock(bucket.span_list_lock);
            for (size_t i = pushed; i < local_count; ++i) {
                void* obj = local_ptrs[i];
                // Heavy operations: Radix Tree lookup and Bitmap modification
                auto* span = PageMap::GetSpan(obj);
                if (!span) {
                    continue;
                }

                span->FreeObject(obj);
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
            }
        }//

    }// end while(cur)
}

void CentralCache::Reset() noexcept {
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        auto& bucket = buckets_[i];
        // ===================================================================
        // 1. 清空 TransferCache (Fast Path 缓存)
        // ===================================================================
        void* head = nullptr;
        bucket.transfer_cache_lock.lock();
        for (size_t j = 0; j < bucket.transfer_cache_count; ++j) {
            void* obj = bucket.transfer_cache[j];
            static_cast<FreeBlock*>(obj)->next = static_cast<FreeBlock*>(head);
            head = obj;
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
            void* cur = head;
            while (cur) {
                void* next = static_cast<FreeBlock*>(cur)->next;
                if (auto* span = PageMap::GetSpan(cur)) {
                    span->FreeObject(cur);
                }
                cur = next;
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
            total_ptrs += kCapScale * batch_num;
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
        // Strategy: TransferCache capacity is 8x the batch size to provide a high-water mark buffer.
        total_ptrs += kCapScale * batch_num;
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
        buckets_[i].transfer_cache_capacity = batch_num * kCapScale;
        buckets_[i].transfer_cache = cur_ptr;
        cur_ptr += buckets_[i].transfer_cache_capacity;
    }
}

Span* CentralCache::GetOneSpan(Bucket& bucket, size_t size, std::unique_lock<std::mutex>& lock) {
    lock.unlock();
    auto page_num = SizeClass::GetMovePageNum(size);
    auto* span = PageCache::GetInstance().AllocSpan(page_num);
    if (!span) {
        return nullptr;
    }

    span->Init(size);
    lock.lock();
    bucket.span_list.push_front(span);
    return span;
}

}// namespace aethermind
