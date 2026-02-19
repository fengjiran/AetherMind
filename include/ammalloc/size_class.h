//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_AMMALLOC_SIZE_CLASS_H
#define AETHERMIND_AMMALLOC_SIZE_CLASS_H

#include "ammalloc/config.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

namespace aethermind {

namespace details {

constexpr static size_t CalculateIndex(size_t size) noexcept {
    if (size == 0) {
        return 0;
    }

    // Fast path for small objects: 8-byte alignment (0-128 bytes)
    // Maps [1, 8] -> 0, ..., [121, 128] -> 15
    if (size <= 128) AM_LIKELY {
            return (size - 1) >> 3;
        }

    /*
     * Stepped Mapping for objects > 128B:
     * 1. msb: Find the power of 2 group (e.g., 129-256B falls into the 2^7 group).
     * 2. group_idx: Normalize msb so that the first group starts at index 0.
     * 3. base_idx: Calculate the starting index of the group.
     * 4. group_offset: Subdivide each power-of-2 group into 2^kStepShift steps.
     */
    int msb = std::bit_width(size - 1) - 1;
    int group_idx = msb - 7;
    size_t base_idx = 16 + (group_idx << SizeConfig::kStepShift);
    int shift = msb - SizeConfig::kStepShift;
    size_t group_offset = ((size - 1) >> shift) & (SizeConfig::kStepsPerGroup - 1);

    return base_idx + group_offset;
}

constexpr static size_t CalculateSize(size_t idx) noexcept {
    // Fast path for small objects (0-128 bytes): Maps index 0..15 back to 8..128
    if (idx < 16) AM_LIKELY {
            return (idx + 1) << 3;
        }

    // Decoding logarithmic stepped index
    size_t relative_idx = idx - 16;
    // Identify the binary group (2^7, 2^8, ...) and the step within it
    size_t group_idx = relative_idx >> SizeConfig::kStepShift;
    size_t step_idx = relative_idx & (SizeConfig::kStepsPerGroup - 1);
    // Reconstruct size components using 64-bit safe shifts
    size_t msb = group_idx + 7;
    size_t base_size = 1ULL << msb;
    size_t step_size = 1ULL << (msb - SizeConfig::kStepShift);
    // Return the upper bound of the current size class ladder
    return base_size + (step_idx + 1) * step_size;
}

}// namespace details

// Validate Small Object Boundaries
static_assert(details::CalculateSize(0) == 8);
static_assert(details::CalculateSize(15) == 128);

// Validate Large Object Group 0 (Range: 129-256)
// Step size = (256-128)/4 = 32
static_assert(details::CalculateSize(16) == 160);// 128 + 32
static_assert(details::CalculateSize(17) == 192);// 160 + 32
static_assert(details::CalculateSize(19) == 256);// Last bucket of group 0

// Validate Large Object Group 1 (Range: 257-512)
// Step size = (512-256)/4 = 64
static_assert(details::CalculateSize(20) == 320);// 256 + 64

// Validate Inverse Property (Index -> Size -> Index)
static_assert(details::CalculateIndex(1) == 0);
static_assert(details::CalculateIndex(8) == 0);
static_assert(details::CalculateIndex(9) == 1);
static_assert(details::CalculateIndex(128) == 15);
static_assert(details::CalculateIndex(129) == 16);// 129 落在 160 的桶里
static_assert(details::CalculateIndex(160) == 16);

/**
 * @brief Static utility class for managing size classes and alignment policies.
 *
 * This class encapsulates all logic related to:
 * 1. Mapping user requested sizes to specific bucket indices (Size Classes).
 * 2. Determining the alignment/capacity of specific buckets.
 * 3. Calculating batch movement strategies between different layers of the memory pool.
 *
 * The alignment strategy follows the Google TCMalloc algorithm:
 * - [1, 128] bytes: 8-byte alignment.
 * - [129, ...] bytes: Exponentially increasing alignment granularity to keep
 *   internal fragmentation low (typically < 12.5%).
 */
class SizeClass {
public:
    /**
     * @brief Maps a requested memory size to its corresponding size class index.
     *
     * This function implements a hybrid mapping strategy to balance memory overhead
     * and lookup speed:
     * 1. Linear Mapping (0 - 128B): Precise 8-byte alignment for the most frequent
     *    small allocations.
     * 2. Logarithmic Stepped Mapping (128B+): Uses a geometric progression (groups)
     *    to maintain a constant relative fragmentation (~12.5% to 25% depending on
     *    kStepShift) while significantly reducing the number of FreeLists in ThreadCache.
     *
     * @param size The requested allocation size in bytes.
     * @return The zero-based index of the size class, or std::numeric_limits<size_t>::max()
     *         if the size is invalid or exceeds MAX_TC_SIZE.
     *
     * @note This implementation is branch-prediction friendly and utilizes C++20
     *       bit-manipulation (std::bit_width) for O(1) performance without large tables.
     */
    constexpr static size_t Index(size_t size) noexcept {
        // clang-format off
        if (size > SizeConfig::MAX_TC_SIZE) AM_UNLIKELY {
            return std::numeric_limits<size_t>::max();
        }

        // Fast path: O(1) table lookup for small objects
        if (size <= SizeConfig::kSmallSizeThreshold) AM_LIKELY {
            return small_index_table_[size];
        }
        // clang-format on

        // Slow path: Mathematical calculation for large objects
        return details::CalculateIndex(size);
    }

    /**
     * @brief Reconstructs the maximum object size for a given size class index.
     *
     * This function serves as the exact inverse of Index.
     * It decodes the logical index back into the actual byte size of the memory block.
     *
     * ### Mathematical Inverse Model
     *
     * 1. **Linear Range** ($idx < 16$):
     *    The size is recovered using a constant 8-byte stride:
     *    $S = (idx + 1) \times 8$
     *
     * 2. **Log-Stepped Range** ($idx \ge 16$):
     *    The function decodes the group and step components:
     *    - **Group Identification**: $msb = \lfloor (idx - 16) / 2^k \rfloor + 7$
     *      (Determines the power-of-2 interval, e.g., 128-256, 256-512, etc.)
     *    - **Step Identification**: $step\_idx = (idx - 16) \pmod{2^k}$
     *      (Determines the subdivision within the power-of-2 interval.)
     *    - **Size Recovery**: $S = 2^{msb} + (step\_idx + 1) \times 2^{msb-k}$
     *
     * This ensures that Size(Index(s)) \ge s$
     * for any $s \in (0, MAX\_TC\_SIZE]$.
     *
     * @param idx The size class index to be decoded.
     * @return The maximum byte size of the objects stored in this size class's FreeList.
     */
    static constexpr size_t Size(size_t idx) noexcept {
        // O(1) table lookup for all size classes
        return size_table_[idx];
    }

    /**
     * @brief Rounds up the requested size to the nearest aligned size class.
     * @param size User requested size.
     * @return Aligned size.
     */
    static constexpr size_t RoundUp(size_t size) noexcept {
        if (size > SizeConfig::MAX_TC_SIZE) AM_UNLIKELY {
                return size;
            }

        return Size(Index(size));
    }

    // -----------------------------------------------------------------------
    // Batch Movement Strategies
    // -----------------------------------------------------------------------

    /**
     * @brief Calculates the batch size for moving objects between ThreadCache and CentralCache.
     *
     * This strategy balances lock contention and memory usage:
     * - Small objects: Move more objects (up to 512) to amortize the cost of locking CentralCache.
     * - Large objects: Move fewer objects (down to 2) to prevent ThreadCache from hoarding memory.
     *
     * @param size The size of the object.
     * @return size_t Number of objects to move in one batch.
     */
    static constexpr size_t CalculateBatchSize(size_t size) noexcept {
        if (size == 0) AM_UNLIKELY {
                return 0;
            }

        // Base strategy: Inverse proportion to size.
        // Example: 32KB / 8B = 4096 (Clamped to 512).
        // Example: 32KB / 32KB = 1 (Clamped to 2).
        size_t batch = SizeConfig::MAX_TC_SIZE / size;
        // Lower bound: Always move at least 2 objects to leverage cache locality.
        if (batch < 2) {
            batch = 2;
        }

        // At most 512, to prevent the central cache pool from being drained instantly
        // Upper bound: Cap at 512 to prevent CentralCache depletion and excessive ThreadCache footprint.
        if (batch > 512) {
            batch = 512;
        }

        return batch;
    }

    /**
     * @brief Calculates the number of pages CentralCache should request from PageCache.
     *
     * This strategy determines the size of the Span (in pages) allocated by CentralCache.
     * It ensures that a single Span can satisfy multiple batch requests from ThreadCache,
     * reducing the frequency of accessing the global PageCache lock.
     *
     * @param size The size of the object.
     * @return size_t Number of pages to allocate (1 to MAX_PAGE_NUM).
     */
    static constexpr size_t GetMovePageNum(size_t size) noexcept {
        // 1. Get the batch size used by ThreadCache.
        size_t batch_num = CalculateBatchSize(size);

        // 2. Amortization Goal:
        // We want the Span to hold enough objects for approximately 8 batch transfers.
        size_t total_objs = batch_num << 3;
        // 3. Convert total bytes to pages.
        size_t total_bytes = total_objs * size;
        // Optimization: For tiny objects, ensure we allocate at least 32KB (8 pages)
        // to minimize metadata overhead (Span structure + Bitmap) per object.
        if (total_bytes < 32 * 1024) {
            total_bytes = 32 * 1024;
        }

        size_t page_num = (total_bytes + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
        // 4. Boundary Enforcement
        if (page_num < 1) {
            page_num = 1;
        }

        if (page_num > PageConfig::MAX_PAGE_NUM) {
            page_num = PageConfig::MAX_PAGE_NUM;
        }

        return page_num;
    }

    SizeClass() = delete;

    /**
     * @brief The total number of size classes (buckets) available.
     * Calculated at compile-time to size the arrays in ThreadCache/CentralCache.
     */
    constexpr static size_t kNumSizeClasses = details::CalculateIndex(SizeConfig::MAX_TC_SIZE) + 1;

private:
    // -----------------------------------------------------------------------
    // Compile-time Lookup Tables (IILE)
    // -----------------------------------------------------------------------

    // Table for O(1) Index lookup (Size -> Index)
    // Only covers small objects up to kSmallSizeThreshold
    constexpr static auto small_index_table_ = []() consteval {
        std::array<uint8_t, SizeConfig::kSmallSizeThreshold + 1> small_index_table{};
        for (size_t sz = 0; sz <= SizeConfig::kSmallSizeThreshold; ++sz) {
            small_index_table[sz] = static_cast<uint8_t>(details::CalculateIndex(sz));
        }
        return small_index_table;
    }();

    // Table for O(1) Size lookup (Index -> Size)
    // Covers ALL indices
    constexpr static auto size_table_ = []() consteval {
        std::array<uint32_t, kNumSizeClasses> size_table{};
        for (size_t idx = 0; idx < kNumSizeClasses; ++idx) {
            size_table[idx] = static_cast<uint32_t>(details::CalculateSize(idx));
        }
        return size_table;
    }();
};

static_assert(SizeClass::Size(0) == 8);
static_assert(SizeClass::Size(15) == 128);
static_assert(SizeClass::Size(16) == 160);
static_assert(SizeClass::Size(19) == 256);
static_assert(SizeClass::Size(20) == 320);
// Round-trip check
static_assert(SizeClass::Index(SizeClass::Size(20)) == 20);
static_assert(SizeClass::Index(129) == 16);
static_assert(SizeClass::Index(150) == 16);

}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_SIZE_CLASS_H
