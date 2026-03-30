// Copyright 2026 The AetherMind Authors
// SPDX-License-Identifier: Apache-2.0
//
// Thread-local front-end cache for ammalloc small-object allocation.
//
// Design goals:
// - Keep the hot allocation/deallocation path completely lock-free.
// - Amortize CentralCache lock traffic through batched refill/trim operations.
// - Limit per-thread memory hoarding with slow-start growth and overages-based decay.
//
// Thread-safety:
// - A ThreadCache instance is intended to be owned by one thread only.
// - Its FreeLists are not synchronized and must never be shared across threads.
// - Cross-thread balancing happens only through CentralCache slow paths.

#ifndef AETHERMIND_AMMALLOC_THREAD_CACHE_H
#define AETHERMIND_AMMALLOC_THREAD_CACHE_H

#include "ammalloc/central_cache.h"

namespace aethermind {

/// Per-thread cache for thread-cacheable size classes.
///
/// ThreadCache provides the allocator's lowest-latency path: a TLS-owned array
/// of LIFO FreeLists indexed by size class. The common case never takes a lock
/// and never touches global metadata.
///
/// Key design properties:
/// - Fast path stays entirely thread-local.
/// - Slow-start grows each FreeList only after repeated refill pressure.
/// - Overflow deallocation trims one batch back to CentralCache.
/// - Repeated overflow events decay `max_size` to avoid sticky high-water marks
///   after transient bursts.
///
/// Lifetime model:
/// - Objects cached here remain owned by the allocator system.
/// - `ReleaseAll()` drains all thread-local state back to CentralCache.
class alignas(SystemConfig::CACHE_LINE_SIZE) ThreadCache {
public:
    ThreadCache() noexcept = default;

    /// Thread-local caches are bound to one thread and are not movable/copyable.
    ThreadCache(const ThreadCache&) = delete;
    ThreadCache& operator=(const ThreadCache&) = delete;

    /// Allocates one object from the FreeList of `aligned_size`.
    ///
    /// @param aligned_size Size-class-aligned object size. Callers must pass the
    ///        internal aligned size, not the original user request.
    /// @return Pointer to an object slot, or nullptr if the slow path cannot
    ///         refill from CentralCache.
    ///
    /// @pre `aligned_size <= SizeConfig::MAX_TC_SIZE`
    /// @note The fast path is a single FreeList pop with no locking.
    AM_NODISCARD AM_ALWAYS_INLINE void* Allocate(size_t aligned_size) noexcept {
        AM_DCHECK(aligned_size <= SizeConfig::MAX_TC_SIZE);
        size_t idx = SizeClass::Index(aligned_size);
        auto& list = free_lists_[idx];

        // Hot path: satisfy the request entirely from TLS state.
        // clang-format off
        if (!list.empty()) AM_LIKELY {
            return list.pop();
        }
        // clang-format on

        // Refill from CentralCache only after local capacity is exhausted.
        return FetchFromCentralCache(list, aligned_size);
    }

    /// Returns one object to the TLS cache for its size class.
    ///
    /// @param ptr Object pointer being freed.
    /// @param aligned_size Span-recorded aligned object size.
    ///
    /// @pre `ptr != nullptr`
    /// @pre `aligned_size <= SizeConfig::MAX_TC_SIZE`
    /// @note The fast path is a single FreeList push. Slow path is entered only
    ///       when local occupancy reaches the current per-class limit.
    void AM_ALWAYS_INLINE Deallocate(void* ptr, size_t aligned_size) {
        AM_DCHECK(ptr != nullptr);
        AM_DCHECK(aligned_size <= SizeConfig::MAX_TC_SIZE);

        size_t idx = SizeClass::Index(aligned_size);
        auto& list = free_lists_[idx];

        // Hot path: keep recently freed objects local to preserve locality.
        list.push(ptr);

        // Crossing the local quota triggers batched trim back to CentralCache.
        // clang-format off
        if (list.size() >= list.max_size()) AM_UNLIKELY {
            DeallocateSlowPath(list, aligned_size);
        }
        // clang-format on
    }

    /// Drains every size-class FreeList back to CentralCache.
    ///
    /// Used during TLS teardown and tests to avoid keeping thread-local state
    /// alive longer than the owning thread.
    void ReleaseAll();

    /// Test-only introspection hook for a FreeList's current high-water limit.
    AM_NODISCARD size_t GetMaxSizeForTest(size_t idx) const noexcept {
        AM_DCHECK(idx < free_lists_.size());
        return free_lists_[idx].max_size();
    }

    /// Test-only introspection hook for the accumulated overage counter.
    AM_NODISCARD size_t GetOveragesForTest(size_t idx) const noexcept {
        AM_DCHECK(idx < free_lists_.size());
        return free_lists_[idx].overages();
    }

private:
    // One TLS-owned LIFO cache per size class. No synchronization is required
    // because the owning thread is the only mutator.
    std::array<FreeList, SizeClass::kNumSizeClasses> free_lists_{};

    /// Refills an empty FreeList from CentralCache and updates its quota.
    ///
    /// The quota follows a two-stage policy:
    /// - exponential warmup until one batch,
    /// - linear growth up to a bounded multiple of the batch size.
    AM_NOINLINE static void* FetchFromCentralCache(FreeList& list, size_t aligned_size);

    /// Trims one batch back to CentralCache and applies overages-based decay.
    ///
    /// Repeated overflow trims without intervening refill demand reduce
    /// `max_size`, preventing long-lived threads from pinning burst-era quotas.
    AM_NOINLINE static void DeallocateSlowPath(FreeList& list, size_t aligned_size);
};
}// namespace aethermind

#endif// AETHERMIND_AMMALLOC_THREAD_CACHE_H
