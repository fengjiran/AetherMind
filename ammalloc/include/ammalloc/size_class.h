//
// ammalloc/size_class.h
//
// Size Class grading system.
//
// Responsible for mapping arbitrary memory requests to fixed-size buckets,
// controlling internal fragmentation to ~12.5%-25% depending on kStepShift.
// Uses TCMalloc-style size classing:
//   - Small objects: Table lookup O(1)
//   - Large objects: Bitwise operations O(1)
//
// Primary exports:
//   - SizeClass::Index() - Size -> Index
//   - SizeClass::Size() - Index -> Size
//   - SizeClass::RoundUp() - Round up to bucket size
//
// Thread Safety: All methods are pure functions, thread-safe.
// ABI Stability: Bucket sizes and counts are compile-time constants,
//                do not affect ABI.
//

#ifndef AETHERMIND_AMMALLOC_SIZE_CLASS_H
#define AETHERMIND_AMMALLOC_SIZE_CLASS_H

#include "ammalloc/config.h"
#include "utils/logging.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

namespace aethermind {

namespace details {

static constexpr size_t CalculateIndex(size_t original_size) noexcept {
    if (original_size == 0) {
        return 0;
    }

    // Linear mapping for the first size range: 8-byte alignment in [1, 128].
    // Maps [1, 8] -> 0, ..., [121, 128] -> 15.
    // Index() uses table lookup up to kSmallSizeThreshold; this helper only
    // describes the underlying piecewise mapping formula.
    // clang-format off
    if (original_size <= 128) AM_LIKELY {
        return (original_size - 1) >> 3;
    }
    // clang-format on

    // Stepped Mapping for objects > 128B:
    // 1. msb: Find the power of 2 group (e.g., 129-256B falls into the 2^7 group).
    // 2. group_idx: Normalize msb so that the first group starts at index 0.
    // 3. base_idx: Calculate the starting index of the group.
    // 4. group_offset: Subdivide each power-of-2 group into 2^kStepShift steps.
    int msb = std::bit_width(original_size - 1) - 1;
    int group_idx = msb - 7;
    size_t base_idx = 16 + (group_idx << SizeConfig::kStepShift);
    int shift = msb - SizeConfig::kStepShift;
    size_t group_offset = ((original_size - 1) >> shift) & (SizeConfig::kStepsPerGroup - 1);

    return base_idx + group_offset;
}

static constexpr size_t CalculateSize(size_t idx) noexcept {
    // Fast path for small objects (0-128 bytes): Maps index 0..15 back to 8..128
    // clang-format off
    if (idx < 16) AM_LIKELY {
        return (idx + 1) << 3;
    }
    // clang-format on

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
static_assert(details::CalculateIndex(129) == 16);// Falls into 160-byte bucket
static_assert(details::CalculateIndex(160) == 16);

/// Static utility class for managing size classes and alignment policies.
///
/// This class encapsulates all logic related to:
/// 1. Mapping user requested sizes to specific bucket indices (Size Classes).
/// 2. Determining the alignment/capacity of specific buckets.
/// 3. Calculating batch movement strategies between different layers of the memory pool.
///
/// The alignment strategy follows the Google TCMalloc algorithm:
/// - [1, 128] bytes: 8-byte alignment.
/// - [129, ...] bytes: Exponentially increasing alignment granularity to keep
///   internal fragmentation low (~12.5%-25% depending on kStepShift).
class SizeClass {
public:
    /// Maps a requested memory size to its corresponding size class index.
    ///
    /// This function implements a hybrid mapping strategy to balance memory overhead
    /// and lookup speed:
    /// 1. Linear Mapping [1, 128] bytes: Precise 8-byte alignment for the most frequent
    ///    small allocations.
    /// 2. Logarithmic Stepped Mapping (128B+): Uses a geometric progression (groups)
    ///    to maintain a constant relative fragmentation (~12.5% to 25% depending on
    ///    kStepShift) while significantly reducing the number of FreeLists in ThreadCache.
    ///
    /// Special case for size=0:
    /// - `Index(0)` returns 0 (maps to the minimum 8-byte size class).
    /// - This design allows mapping interfaces (Index, RoundUp) to handle size=0 gracefully,
    ///   while strategy interfaces (CalculateBatchSize, GetMovePageNum) treat 0 as invalid input.
    /// - Rationale: `am_malloc(0)` must return a valid pointer (C standard), so the mapping
    ///   layer accommodates it; batch/page calculations don't make sense for zero-sized objects.
    ///
    /// @param original_size The requested allocation size in bytes.
    /// @return The zero-based index of the size class, or std::numeric_limits<size_t>::max()
    ///         if the size exceeds MAX_TC_SIZE.
    ///
    /// @note This implementation is branch-prediction friendly and utilizes C++20
    ///       bit-manipulation (std::bit_width) for O(1) performance. Index() uses
    ///       a precomputed lookup table for sizes in [0, kSmallSizeThreshold], and
    ///       falls back to the arithmetic mapping formula for larger in-range sizes.
    AM_ALWAYS_INLINE static constexpr size_t Index(size_t original_size) noexcept {
        // clang-format off
        if (original_size > SizeConfig::MAX_TC_SIZE) AM_UNLIKELY {
            return std::numeric_limits<size_t>::max();
        }

        // Fast path: O(1) table lookup for small objects
        if (original_size <= SizeConfig::kSmallSizeThreshold) AM_LIKELY {
            return small_index_table_[original_size];
        }
        // clang-format on

        // Slow path: Mathematical calculation for large objects
        return details::CalculateIndex(original_size);
    }

    /// Reconstructs the bucket size for a given size class index.
    ///
    /// This function decodes the logical index back into the actual byte size
    /// of the memory block. It satisfies:
    /// - `Index(Size(idx)) == idx` for all valid indices
    /// - `Size(Index(s)) >= s` for all valid sizes (not strict equality)
    ///
    /// Thus Size is a left-inverse of Index, but not a strict bijection
    /// because Index maps multiple sizes to the same class (e.g., 129→16, 160→16).
    ///
    /// ### Mathematical Inverse Model
    ///
    /// 1. **Linear Range** ($idx < 16$):
    ///    The size is recovered using a constant 8-byte stride:
    ///    $S = (idx + 1) \times 8$
    ///
    /// 2. **Log-Stepped Range** ($idx \ge 16$):
    ///    The function decodes the group and step components:
    ///    - **Group Identification**: $msb = \lfloor (idx - 16) / 2^k \rfloor + 7$
    ///      (Determines the power-of-2 interval, e.g., 128-256, 256-512, etc.)
    ///    - **Step Identification**: $step\_idx = (idx - 16) \pmod{2^k}$
    ///      (Determines the subdivision within the power-of-2 interval.)
    ///    - **Size Recovery**: $S = 2^{msb} + (step\_idx + 1) \times 2^{msb-k}$
    ///
    /// This ensures that $Size(Index(s)) \ge s$
    /// for any $s \in (0, MAX\_TC\_SIZE]$.
    ///
    /// @param idx The size class index to be decoded.
    /// @return The maximum byte size of the objects stored in this size class's FreeList.
    AM_ALWAYS_INLINE static constexpr size_t Size(size_t idx) noexcept {
        // O(1) table lookup for all size classes
        // Caller must ensure idx < kNumSizeClasses
        return size_table_[idx];
    }

    /// Debug/assertion version of Size() with bounds checking.
    ///
    /// This is a debugging/contract checking interface. If idx is out of range,
    /// it triggers AM_CHECK(false) which aborts the process. The returned 0
    /// is unreachable in practice.
    ///
    /// @param idx The size class index.
    /// @return The maximum byte size (only returns on valid idx).
    AM_ALWAYS_INLINE static size_t SafeSize(size_t idx) noexcept {
        // clang-format off
        if (idx >= kNumSizeClasses) AM_UNLIKELY {
            AM_CHECK(false, "SizeClass::Size index {} out of range [0, {})", idx, kNumSizeClasses);
            return 0;
        }
        // clang-format on
        return size_table_[idx];
    }

    /// Rounds up the requested size to the nearest size class boundary.
    ///
    /// Behavior:
    /// - For `size <= MAX_TC_SIZE`: Returns the smallest size class >= size.
    /// - For `size > MAX_TC_SIZE`: Returns `size` unchanged (passthrough).
    ///
    /// The passthrough behavior for oversize values means this function does NOT
    /// guarantee alignment for large allocations outside ThreadCache's scope.
    /// Callers must handle oversize allocations separately (e.g., direct page allocation).
    ///
    /// @param original_size User requested size.
    /// @return Aligned size class, or original size if exceeds MAX_TC_SIZE.
    AM_ALWAYS_INLINE static constexpr size_t RoundUp(size_t original_size) noexcept {
        size_t idx = Index(original_size);
        // clang-format off
        if (idx == std::numeric_limits<size_t>::max()) AM_UNLIKELY {
            return original_size;
        }
        // clang-format on

        return size_table_[idx];
    }

    // -----------------------------------------------------------------------
    // Batch Movement Strategies
    // -----------------------------------------------------------------------

    /// Calculates the batch size for moving objects between ThreadCache and CentralCache.
    ///
    /// This strategy balances lock contention and memory usage.
    /// The policy is defined per size class, not per raw request size: requests
    /// that map to the same class always get the same batch size.
    ///
    /// - Small objects: Move more objects (up to 512) to amortize the cost of locking CentralCache.
    /// - Large objects: Move fewer objects (down to 2) to prevent ThreadCache from hoarding memory.
    ///
    /// @param aligned_size Requested or already-rounded object size.
    /// @return Number of objects to move in one batch, or 0 for invalid input.
    static constexpr size_t CalculateBatchSize(size_t aligned_size) noexcept {
        // clang-format off
        if (aligned_size == 0 || aligned_size > SizeConfig::MAX_TC_SIZE) AM_UNLIKELY {
            return 0;
        }
        // clang-format on

        return BatchByIndex(Index(aligned_size));
    }

    /// Calculates the number of pages CentralCache should request from PageCache.
    ///
    /// This strategy determines the size of the Span (in pages) allocated by CentralCache.
    /// The policy is defined per size class, not per raw request size: requests
    /// that map to the same class always get the same span size.
    ///
    /// It ensures that a single Span can satisfy multiple batch requests from ThreadCache,
    /// reducing the frequency of accessing the global PageCache lock.
    ///
    /// @param aligned_size Requested or already-rounded object size.
    /// @return Number of pages to allocate, or 0 for invalid input.
    AM_ALWAYS_INLINE static constexpr size_t GetMovePageNum(size_t aligned_size) noexcept {
        if (aligned_size == 0 || aligned_size > SizeConfig::MAX_TC_SIZE) return 0;
        return MovePagesByIndex(Index(aligned_size));
    }

    SizeClass() = delete;

    /// The total number of size classes (buckets) available.
    /// Calculated at compile-time to size the arrays in ThreadCache/CentralCache.
    static constexpr size_t kNumSizeClasses = details::CalculateIndex(SizeConfig::MAX_TC_SIZE) + 1;

    static constexpr size_t kMaxBatchSize = 512;

private:
    AM_ALWAYS_INLINE static uint16_t BatchByIndex(size_t idx) noexcept {
        return batch_table_[idx];
    }

    AM_ALWAYS_INLINE static uint16_t MovePagesByIndex(size_t idx) noexcept {
        return move_page_table_[idx];
    }

    // -----------------------------------------------------------------------
    // Compile-time Lookup Tables (IILE)
    // -----------------------------------------------------------------------

    // Table for O(1) Index lookup (Size -> Index)
    // Only covers small objects up to kSmallSizeThreshold
    static constexpr auto small_index_table_ = []() consteval {
        std::array<uint8_t, SizeConfig::kSmallSizeThreshold + 1> small_index_table{};
        for (size_t sz = 0; sz <= SizeConfig::kSmallSizeThreshold; ++sz) {
            small_index_table[sz] = static_cast<uint8_t>(details::CalculateIndex(sz));
        }
        return small_index_table;
    }();

    // Table for O(1) Size lookup (Index -> Size)
    // Covers ALL indices
    static constexpr auto size_table_ = []() consteval {
        std::array<uint32_t, kNumSizeClasses> size_table{};
        for (size_t idx = 0; idx < kNumSizeClasses; ++idx) {
            size_table[idx] = static_cast<uint32_t>(details::CalculateSize(idx));
        }
        return size_table;
    }();

    static constexpr auto batch_table_ = []() consteval {
        std::array<uint16_t, kNumSizeClasses> t{};
        for (size_t idx = 0; idx < kNumSizeClasses; ++idx) {
            size_t norm = size_table_[idx];
            size_t batch = SizeConfig::MAX_TC_SIZE / norm;
            if (batch < 2) {
                batch = 2;
            }
            if (batch > kMaxBatchSize) {
                batch = kMaxBatchSize;
            }
            t[idx] = static_cast<uint16_t>(batch);
        }
        return t;
    }();

    static constexpr auto move_page_table_ = []() consteval {
        std::array<uint16_t, kNumSizeClasses> t{};
        for (size_t idx = 0; idx < kNumSizeClasses; ++idx) {
            size_t norm = size_table_[idx];
            size_t batch = batch_table_[idx];
            size_t total_objs = batch << 3;
            size_t total_bytes = total_objs * norm;
            if (total_bytes < 32 * 1024) {
                total_bytes = 32 * 1024;
            }

            size_t pages = (total_bytes + SystemConfig::PAGE_SIZE - 1) >> SystemConfig::PAGE_SHIFT;
            if (pages > PageConfig::MAX_PAGE_NUM) {
                pages = PageConfig::MAX_PAGE_NUM;
            }

            t[idx] = static_cast<uint16_t>(pages);
        }
        return t;
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

static_assert(SizeClass::Index(0) == 0);
static_assert(SizeClass::RoundUp(0) == 8);
static_assert(SizeClass::kNumSizeClasses <= std::numeric_limits<uint8_t>::max());

// MAX_TC_SIZE must be a power of two to land exactly on a size class boundary.
// With kStepsPerGroup=4, each power-of-2 interval is evenly divided,
// so 2^n (n>=7) is always the upper bound of some size class.
static_assert(std::has_single_bit(SizeConfig::MAX_TC_SIZE),
              "MAX_TC_SIZE must be a power of two to ensure it lands on a size class boundary");
static_assert(SizeClass::Size(SizeClass::kNumSizeClasses - 1) == SizeConfig::MAX_TC_SIZE);

// ===========================================================================
// Compile-time validation for SizeClass invariants
// Uses sampling strategy to avoid constexpr step limit (32K iterations exceeds limit)
// ===========================================================================

namespace details {

// Sample key boundaries: each size class boundary + 1
consteval bool ValidateIndexInRangeSampled() {
    for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
        size_t class_size = SizeClass::Size(idx);
        // Test the class boundary and the value just before it
        if (idx > 0) {
            size_t prev_class = SizeClass::Size(idx - 1);
            // Value at previous class boundary
            if (SizeClass::Index(prev_class) != idx - 1) return false;
            // Value just inside this class
            if (SizeClass::Index(prev_class + 1) != idx) return false;
        }
        // Value at this class boundary
        if (SizeClass::Index(class_size) != idx) return false;
    }
    return true;
}

consteval bool ValidateSizeNotLessThanInputSampled() {
    // Test at each class boundary
    for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
        size_t class_size = SizeClass::Size(idx);
        // At boundary: Size(Index(s)) == s
        if (SizeClass::Size(SizeClass::Index(class_size)) != class_size) return false;
        // Just inside class: Size(Index(s)) > s
        if (idx > 0) {
            size_t prev_class = SizeClass::Size(idx - 1);
            size_t mid = (prev_class + class_size) / 2;
            if (SizeClass::Size(SizeClass::Index(mid)) < mid) return false;
        }
    }
    return true;
}

consteval bool ValidateIndexIdempotentSampled() {
    for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
        size_t class_size = SizeClass::Size(idx);
        // Index(Size(Index(s))) == Index(s) at boundaries
        if (SizeClass::Index(SizeClass::Size(SizeClass::Index(class_size))) != SizeClass::Index(class_size)) {
            return false;
        }
    }
    return true;
}

// Full validation: only 48 indices, fits within constexpr limit
consteval bool ValidateSizeMonotonic() {
    for (size_t idx = 1; idx < SizeClass::kNumSizeClasses; ++idx) {
        if (SizeClass::Size(idx) <= SizeClass::Size(idx - 1)) return false;
    }
    return true;
}

consteval bool ValidateRoundUpMonotonicSampled() {
    // Test at each class boundary
    size_t prev = SizeClass::RoundUp(1);
    for (size_t idx = 0; idx < SizeClass::kNumSizeClasses; ++idx) {
        size_t class_size = SizeClass::Size(idx);
        size_t curr = SizeClass::RoundUp(class_size);
        if (curr < prev) return false;
        prev = curr;
    }
    return true;
}

}// namespace details

static_assert(details::ValidateIndexInRangeSampled(), "Index(s) must map to valid class at boundaries");
static_assert(details::ValidateSizeNotLessThanInputSampled(), "Size(Index(s)) must be >= s at class boundaries");
static_assert(details::ValidateIndexIdempotentSampled(), "Index(Size(Index(s))) must equal Index(s) at boundaries");
static_assert(details::ValidateSizeMonotonic(), "Size(idx) must be strictly increasing");
static_assert(details::ValidateRoundUpMonotonicSampled(), "RoundUp(s) must be non-decreasing at class boundaries");
}// namespace aethermind

#endif// AETHERMIND_AMMALLOC_SIZE_CLASS_H
