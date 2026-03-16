/// Page-level OS allocator and object pool for the ammalloc subsystem.
///
/// This header defines:
/// - PageAllocator: OS memory mapping facade (mmap/munmap/madvise) with huge-page support
/// - ObjectPool: Thread-safe pool for fixed-size objects backed by PageAllocator pages
///
/// Thread-safety:
/// - All public methods are thread-safe.
/// - Huge-page cache uses internal mutex synchronization.
///
/// Constraints:
/// - Core allocation paths avoid recursive malloc (no heap-allocating containers).
/// - Allocation failures return nullptr; no exceptions thrown (except ObjectPool::New).
///
/// @see PageCache, am_malloc, am_free

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
    // Statistics are best-effort telemetry only; all updates use relaxed atomics.
    // normal page
    std::atomic<size_t> normal_alloc_count{0};
    std::atomic<size_t> normal_alloc_success{0};
    std::atomic<size_t> normal_alloc_bytes{0};

    // huge page
    std::atomic<size_t> huge_alloc_count{0};
    std::atomic<size_t> huge_alloc_success{0};
    std::atomic<size_t> huge_alloc_bytes{0};
    std::atomic<size_t> huge_align_waste_bytes{0};
    std::atomic<size_t> huge_cache_hit_count{0};
    std::atomic<size_t> huge_cache_miss_count{0};

    std::atomic<size_t> free_count{0};
    std::atomic<size_t> free_bytes{0};

    std::atomic<size_t> normal_alloc_failed_count{0};
    std::atomic<size_t> huge_alloc_failed_count{0};
    std::atomic<size_t> alloc_failed_count{0};
    std::atomic<size_t> huge_fallback_to_normal_count{0};
    std::atomic<size_t> mmap_enomem_count{0};
    std::atomic<size_t> mmap_other_error_count{0};
    std::atomic<size_t> munmap_failed_count{0};
    std::atomic<size_t> madvise_failed_count{0};
};

/// Page-level OS allocator used by PageCache/ObjectPool backends.
///
/// Thread-safety:
/// - Public methods are thread-safe.
/// - Huge-page cache is internally synchronized with a mutex.
///
/// Notes:
/// - Calls can block in kernel (`mmap`/`munmap`/`madvise`).
/// - Returns `nullptr` on allocation failure; does not throw.
class PageAllocator {
public:
    static const PageAllocatorStats& GetStats() {
        return stats_;
    }

    /// Allocates `page_num` pages and returns a page-aligned pointer.
    /// Returns nullptr on invalid input or system allocation failure.
    static void* SystemAlloc(size_t page_num);

    /// Frees a mapping previously returned by `SystemAlloc`.
    /// `ptr == nullptr` or `page_num == 0` is treated as a no-op.
    static void SystemFree(void* ptr, size_t page_num);

    /// Resets all statistics counters to zero.
    static void ResetStats();

    /// Releases all cached huge pages.
    /// Intended for tests and controlled teardown paths.
    static void ReleaseHugePageCache();

    /// Records a munmap failure in statistics.
    /// Used internally by HugePageCache during cache release.
    static void RecordMunmapFailure() {
        stats_.munmap_failed_count.fetch_add(1, std::memory_order_relaxed);
    }

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

/// Thread-safe object pool backed by `PageAllocator` pages.
///
/// The pool owns all allocated chunks until `ReleaseMemory()` or destruction.
/// `New()` constructs `T` in-place and `Delete()` destroys `T` then recycles storage.
template<typename T, size_t CHUNK_SIZE = 64 * 1024>
    requires(sizeof(T) >= sizeof(void*) && std::default_initializable<T>)
class ObjectPool {
public:
    ObjectPool() = default;

    /// Allocates storage for one object and default-constructs `T` in-place.
    /// Throws `std::bad_alloc` if the underlying page allocation fails.
    template<typename... Args>
    T* New(Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_) {
            void* obj = free_list_;
            free_list_ = free_list_->next;
            return new (obj) T(std::forward<Args>(args)...);
        }

        if (remain_bytes_ < sizeof(T)) {
            size_t num_objs = CHUNK_SIZE / sizeof(T);
            if (num_objs < 10) {
                num_objs = 10;
            }
            size_t needed_bytes = sizeof(ChunkHeader) + num_objs * sizeof(T) + alignof(T) - 1;
            size_t page_num = (needed_bytes + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
            void* ptr = PageAllocator::SystemAlloc(page_num);
            if (!ptr) {
                throw std::bad_alloc();
            }

            auto* new_chunk = static_cast<ChunkHeader*>(ptr);
            new_chunk->next = chunk_list_;
            new_chunk->page_num = page_num;
            chunk_list_ = new_chunk;

            uintptr_t raw_data_start = reinterpret_cast<uintptr_t>(ptr) + sizeof(ChunkHeader);
            uintptr_t aligned_data_start = (raw_data_start + alignof(T) - 1) & (~(alignof(T) - 1));
            data_ = reinterpret_cast<char*>(aligned_data_start);

            size_t total_bytes = page_num << SystemConfig::PAGE_SHIFT;
            remain_bytes_ = total_bytes - (aligned_data_start - reinterpret_cast<uintptr_t>(ptr));
        }

        void* obj = data_;
        data_ += sizeof(T);
        remain_bytes_ -= sizeof(T);
        return new (obj) T(std::forward<Args>(args)...);
    }

    /// Destroys `obj` and returns its storage to the pool free list.
    void Delete(T* obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        obj->~T();
        auto* header = reinterpret_cast<FreeHeader*>(obj);
        header->next = free_list_;
        free_list_ = header;
    }

    /// Releases all chunks owned by this pool.
    /// Callers must ensure no outstanding objects are used afterwards.
    void ReleaseMemory() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* cur = chunk_list_;
        while (cur) {
            auto* next = cur->next;
            PageAllocator::SystemFree(cur, cur->page_num);
            cur = next;
        }
        chunk_list_ = nullptr;
        free_list_ = nullptr;
        data_ = nullptr;
        remain_bytes_ = 0;
    }

    ~ObjectPool() {
        auto* cur = chunk_list_;
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
    ChunkHeader* chunk_list_{nullptr};
    std::mutex mutex_;
};

}// namespace aethermind

#endif// AETHERMIND_MALLOC_PAGE_ALLOCATOR_H
