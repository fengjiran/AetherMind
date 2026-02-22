//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_AMMALLOC_THREAD_CACHE_H
#define AETHERMIND_AMMALLOC_THREAD_CACHE_H

#include "ammalloc/central_cache.h"
#include <cstddef>

namespace aethermind {

/**
 * @brief Per-thread memory cache (TLS) for high-speed allocation.
 *
 * ThreadCache is the "Frontend" of the memory pool. It is lock-free and
 * handles the vast majority of malloc/free requests (Fast Path).
 * Only communicates with CentralCache (Slow Path) when empty or full.
 */
class alignas(SystemConfig::CACHE_LINE_SIZE) ThreadCache {
public:
    ThreadCache() noexcept = default;

    // Disable copy/move (TLS objects shouldn't be moved)
    ThreadCache(const ThreadCache&) = delete;
    ThreadCache& operator=(const ThreadCache&) = delete;

    /**
     * @brief Allocate memory of a specific size.
     * @param size User requested size (must be <= MAX_TC_SIZE).
     * @return Pointer to the allocated memory.
     */
    AM_NODISCARD AM_ALWAYS_INLINE void* Allocate(size_t size) noexcept {
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

    /**
     * @brief Deallocate memory.
     * @param ptr Pointer to the memory.
     * @param size The size of the object (lookup via PageMap in global interface).
     */
    void AM_ALWAYS_INLINE Deallocate(void* ptr, size_t size) {
        AM_DCHECK(ptr != nullptr);
        AM_DCHECK(size <= SizeConfig::MAX_TC_SIZE);

        size_t idx = SizeClass::Index(size);
        auto& list = free_lists_[idx];
        // 1. Fast Path: Push to local free list (Lock-Free)
        list.push(ptr);

        // 2. Slow Path: Return memory if cache is too large (Scavenging)
        // If the list length exceeds the limit, return a batch to CentralCache.
        // clang-format off
        if (list.size() >= list.max_size()) AM_UNLIKELY {
            DeallocateSlowPath(list, size);
        }
        // clang-format on
    }

    // Release all memory in ThreadCache to CentralCache.
    void ReleaseAll();

private:
    // Array of FreeLists. Access is lock-free as it's thread-local.
    std::array<FreeList, SizeClass::kNumSizeClasses> free_lists_{};

    /**
     * @brief Fetch objects from CentralCache when ThreadCache is empty.
     */
    AM_NOINLINE static void* FetchFromCentralCache(FreeList& list, size_t size);

    /**
     * @brief Return objects to CentralCache when ThreadCache is full.
     */
    AM_NOINLINE static void DeallocateSlowPath(FreeList& list, size_t size);
};
}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_THREAD_CACHE_H
