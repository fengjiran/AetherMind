//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
#define AETHERMIND_MALLOC_PAGE_ALLOCATOR_H

#include "ammalloc/config.h"

#include <atomic>
#include <cstddef>
#include <mutex>

namespace aethermind {

#ifdef AMMALLOC_TEST
extern std::atomic<bool> g_mock_huge_alloc_fail;
#define PAGEALLOCATOR_FRIEND_TEST   \
    friend class PageAllocatorTest; \
    friend class PageAllocatorThreadSafeTest;
#else
#define g_mock_huge_alloc_fail (false)
#define PAGEALLOCATOR_FRIEND_TEST
#endif

struct PageAllocatorStats {
    // 基础分配统计
    // normal page
    std::atomic<size_t> normal_alloc_count{0};  // 普通页分配请求数
    std::atomic<size_t> normal_alloc_success{0};// 普通页分配成功数
    std::atomic<size_t> normal_alloc_bytes{0};  // 普通页总分配字节数

    // huge page
    std::atomic<size_t> huge_alloc_count{0};      // 大页分配请求数
    std::atomic<size_t> huge_alloc_success{0};    // 大页分配成功数
    std::atomic<size_t> huge_alloc_bytes{0};      // 大页总分配字节数
    std::atomic<size_t> huge_align_waste_bytes{0};// 大页对齐浪费的内存字节数
    std::atomic<size_t> huge_cache_hit_count{0};  // 大页缓存命中数
    std::atomic<size_t> huge_cache_miss_count{0}; // 大页缓存未命中数

    // 释放统计
    std::atomic<size_t> free_count{0};
    std::atomic<size_t> free_bytes{0};

    // 错误统计
    std::atomic<size_t> normal_alloc_failed_count{0};    // 普通页分配失败次数
    std::atomic<size_t> huge_alloc_failed_count{0};      // 大页分配失败次数
    std::atomic<size_t> alloc_failed_count{0};           // 最终分配失败总数
    std::atomic<size_t> huge_fallback_to_normal_count{0};// 大页降级到普通页的次数
    std::atomic<size_t> mmap_enomem_count{0};            // mmap ENOMEM失败次数
    std::atomic<size_t> mmap_other_error_count{0};       // mmap其他错误次数
    std::atomic<size_t> munmap_failed_count{0};          // munmap失败次数
    std::atomic<size_t> madvise_failed_count{0};         // madvise失败次数
};

class PageAllocator {
public:
    static const PageAllocatorStats& GetStats() {
        return stats_;
    }

    static void* SystemAlloc(size_t page_num);
    static void SystemFree(void* ptr, size_t page_num);
    static void ResetStats();
    static void ReleaseHugePageCache();

private:
    inline static PageAllocatorStats stats_;

    static void* AllocWithRetry(size_t size, int flags);
    static void ApplyHugePageHint(void* ptr, size_t size);
    static void* AllocNormalPage(size_t size);
    static void* AllocHugePageWithTrim(size_t size);
    static void* AllocHugePage(size_t size);
    static bool SafeMunmap(void* ptr, size_t size);

    PAGEALLOCATOR_FRIEND_TEST;
};

template<typename T, size_t CHUNK_SIZE = 64 * 1024>
    requires(sizeof(T) >= sizeof(void*) && std::default_initializable<T>)
class ObjectPool {
public:
    ObjectPool() = default;

    T* New() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_) {
            void* obj = free_list_;
            free_list_ = free_list_->next;
            return new (obj) T();
        }

        if (remain_bytes_ < sizeof(T)) {
            size_t num_objs = CHUNK_SIZE / sizeof(T);
            if (num_objs < 10) {
                num_objs = 10;
            }
            size_t needed_bytes = sizeof(ChunkHeader) + num_objs * sizeof(T);
            size_t page_num = (needed_bytes + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
            void* ptr = PageAllocator::SystemAlloc(page_num);
            if (!ptr) {
                throw std::bad_alloc();
            }

            auto* new_chunk = static_cast<ChunkHeader*>(ptr);
            new_chunk->next = chunk_header_;
            new_chunk->page_num = page_num;
            chunk_header_ = new_chunk;

            data_ = static_cast<char*>(ptr) + sizeof(ChunkHeader);
            size_t total_bytes = page_num << SystemConfig::PAGE_SHIFT;
            remain_bytes_ = total_bytes - sizeof(ChunkHeader);
        }

        void* obj = data_;
        data_ += sizeof(T);
        remain_bytes_ -= sizeof(T);
        return new (obj) T();
    }

    void Delete(T* obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        obj->~T();
        auto* header = reinterpret_cast<FreeHeader*>(obj);
        header->next = free_list_;
        free_list_ = header;
    }

    ~ObjectPool() {
        auto* cur = chunk_header_;
        while (cur) {
            auto* next = cur->next;
            PageAllocator::SystemFree(cur, cur->page_num);
            cur = next;
        }
    }

private:
    struct FreeHeader {
        FreeHeader* next;
    };

    struct ChunkHeader {
        ChunkHeader* next;
        size_t page_num;
    };

    char* data_{nullptr};
    size_t remain_bytes_{0};
    FreeHeader* free_list_{nullptr};
    ChunkHeader* chunk_header_{nullptr};
    std::mutex mutex_;
};

}// namespace aethermind

#endif//AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
