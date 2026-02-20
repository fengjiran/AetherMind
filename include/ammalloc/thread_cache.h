//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_AMMALLOC_MEMORY_POOL_H
#define AETHERMIND_AMMALLOC_MEMORY_POOL_H

#include "ammalloc/central_cache.h"

namespace aethermind {

/**
 * @brief Per-thread memory cache (TLS) for high-speed allocation.
 *
 * ThreadCache is the "Frontend" of the memory pool. It is lock-free and
 * handles the vast majority of malloc/free requests (Fast Path).
 * Only communicates with CentralCache (Slow Path) when empty or full.
 */
class alignas(64) ThreadCache {
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
    AM_NODISCARD void* Allocate(size_t size) noexcept;

    /**
     * @brief Deallocate memory.
     * @param ptr Pointer to the memory.
     * @param size The size of the object (lookup via PageMap in global interface).
     */
    void Deallocate(void* ptr, size_t size);

    void ReleaseAll();

private:
    // Array of FreeLists. Access is lock-free as it's thread-local.
    std::array<FreeList, SizeClass::kNumSizeClasses> free_lists_{};

    /**
     * @brief Fetch objects from CentralCache when ThreadCache is empty.
     */
    static void* FetchFromCentralCache(FreeList& list, size_t size);

    /**
     * @brief Return objects to CentralCache when ThreadCache is full.
     */
    static void ReleaseTooLongList(FreeList& list, size_t size);
};
}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_MEMORY_POOL_H
