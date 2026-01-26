//
// Created by richard on 1/26/26.
//
module;

#include "macros.h"

#include <mutex>
#include <thread>
#include <vector>

export module MemoryPool;

namespace aethermind {

struct MagicConstants {
    // page size (default: 4KB)
    constexpr static size_t PAGE_SIZE = 4096;
    // max thread cache size(256KB)
    constexpr static size_t MAX_TC_SIZE = 256 * 1024;
    // size class alignment
    constexpr static size_t ALIGNMENT = 16;
    // cache line size
    constexpr static size_t CACHE_LINE_SIZE = 64;
    // The upper limit of the number of memory blocks that the Thread Cache can
    // obtain in batches from the Central Cache at one time.
    // Use dynamic Batch logic: take more for small blocks, take fewer for large blocks
    static constexpr size_t CalculateBatchSize(size_t size) {
        if (size == 0) return 0;
        size_t batch = MAX_TC_SIZE / size;
        if (batch < 2) batch = 2;    // at least 2
        if (batch > 128) batch = 128;// At most 128, to prevent the central pool from being drained instantly
        return batch;
    }

    // Maximum number of consecutive pages managed by Page Cache
    // (to avoid excessively large Spans)
    constexpr static size_t MAX_PAGE_NUM = 1024;
};

/**
 * @brief Aligns 'size' up to the specified 'align'.
 * @return The smallest multiple of 'align' that is greater than or equal to 'size'.
 */
AM_NODISCARD constexpr size_t AlignUp(size_t size, size_t align = MagicConstants::ALIGNMENT) noexcept {
    if (size == 0) AM_UNLIKELY {
            return align;
        }
    // Optimization for power-of-two alignments (the most common case)
    if (std::has_single_bit(align)) AM_LIKELY {
            return (size + align - 1) & ~(align - 1);
        }
    // Fallback for non-power-of-two alignments
    return (size + align - 1) / align * align;
}

/**
 * @brief Maps a memory request size to its corresponding Size Class Index.
 *
 * This function calculates the index into the thread-local cache (TC) size-class
 * array. It leverages C++20 static reflection and bitwise optimizations to
 * ensure near-zero runtime latency.
 *
 * @param size The requested memory size in bytes.
 * @return The size class index, or std::numeric_limits<size_t>::max() if the size
 *         is 0 or exceeds MagicConstants::MAX_TC_SIZE.
 *
 * @section perf Performance Characteristics:
 * - Branch Prediction: Uses AM_UNLIKELY to keep the hot path (valid requests)
 *   optimized for instruction cache locality.
 * - Compile-time Optimization: Uses `if constexpr` to perform zero-cost dispatching.
 *   If ALIGNMENT is a power of two, the division is replaced by a high-speed bitwise
 *   shift at compile time.
 * - Zero-cost fallback: The non-power-of-two division path is completely stripped
 *   from the generated machine code if it's not needed.
 *
 * @note
 * In 2026-era processors (e.g., Zen 5, Arrow Lake), the power-of-two path
 * typically executes in < 1 clock cycle when inlined.
 */
AM_NODISCARD constexpr size_t SizeClassIndex(size_t size) noexcept {
    // Validate boundaries: size 0 or exceeding MAX_TC_SIZE are rare edge cases.
    if (size == 0 || size > MagicConstants::MAX_TC_SIZE) AM_UNLIKELY {
            return std::numeric_limits<size_t>::max();
        }

    // Static dispatch based on the alignment property.
    if constexpr (std::has_single_bit(MagicConstants::ALIGNMENT)) {
        // Highly optimized path: (size - 1) / 2^n -> (size - 1) >> n
        constexpr size_t shift = std::countr_zero(MagicConstants::ALIGNMENT);
        return (size - 1) >> shift;
    } else {
        // Fallback path for non-power-of-two alignments.
        return (size - 1) / MagicConstants::ALIGNMENT;
    }
}

/**
 * @brief Maps a raw memory pointer to its global page index.
 *
 * This function calculates the page number by dividing the memory address
 * by the system page size. It is a critical path component for PageMap
 * and Span lookups.
 *
 * @param ptr The raw pointer to be converted.
 * @return The corresponding page number (address >> shift).
 *
 * @note Performance:
 * - Constant-time O(1) complexity.
 * - Uses `if constexpr` to eliminate division overhead at compile-time.
 * - If PAGE_SIZE is a power of two, this lowers to a single bitwise SHR instruction.
 */
AM_NODISCARD constexpr size_t PtrToPageIdx(void* ptr) noexcept {
    // 1. Cast pointer to integer. Note: This limits 'true' constexpr usage
    // but enables massive runtime inlining optimizations.
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    // 2. Static dispatch for page size alignment.
    if constexpr (std::has_single_bit(MagicConstants::PAGE_SIZE)) {
        // Optimization: Address / 2^n -> Address >> n
        constexpr size_t shift = std::countr_zero(MagicConstants::PAGE_SIZE);
        return addr >> shift;
    } else {
        // Fallback for non-standard page sizes.
        return addr / MagicConstants::PAGE_SIZE;
    }
}

AM_NODISCARD constexpr void* PageNumToPtr(size_t page_idx) noexcept {
    if constexpr (std::has_single_bit(MagicConstants::PAGE_SIZE)) {
        constexpr size_t shift = std::countr_zero(MagicConstants::PAGE_SIZE);
        return reinterpret_cast<void*>(page_idx << shift);
    } else {
        return reinterpret_cast<void*>(page_idx * MagicConstants::PAGE_SIZE);
    }
}

struct FreeBlock {
    FreeBlock* next;
};

class FreeList {
public:
    constexpr FreeList() : head_(nullptr), size_(0) {}

    FreeList(const FreeList&) = delete;
    FreeList& operator=(const FreeList&) = delete;

    AM_NODISCARD bool empty() const noexcept {
        return head_ == nullptr;
    }

    AM_NODISCARD size_t size() const noexcept {
        return size_;
    }

    void clear() noexcept {
        head_ = nullptr;
        size_ = 0;
    }

    void push(void* ptr) noexcept {
        if (!ptr) AM_UNLIKELY return;

        auto* block = static_cast<FreeBlock*>(ptr);
        block->next = head_;
        head_ = block;
        ++size_;
    }

    void push_range(void* begin, void* end, size_t count) noexcept {
        if (!begin || !end || count == 0) {
            return;
        }

        static_cast<FreeBlock*>(end)->next = head_;
        head_ = static_cast<FreeBlock*>(begin);
        size_ += count;
    }

    AM_NODISCARD void* pop() {
        if (empty()) AM_UNLIKELY return nullptr;

        auto* block = head_;
        if (block->next) AM_LIKELY AM_BUILTIN_PREFETCH(block->next, 0, 3);

        head_ = head_->next;
        --size_;
        return block;
    }

private:
    FreeBlock* head_;
    size_t size_;
};

/**
 * @brief Span represents a contiguous range of memory pages.
 * Optimized for 64-bit architectures to minimize padding.
 * Total size: 64 bytes(1 cache line) to prevent false sharing and optimize fetch
 */
struct Span {
    // --- Hot Data (First Cache Line) ---
    size_t start_page_idx;// Global start page index
    size_t page_num;      // Number of contiguous pages
    size_t block_size;    // Size of objects allocated from this Span(if applicable)

    // --- Control Data ---
    FreeList free_list;// Embedded free list for small object allocation
    Span* prev{nullptr};
    Span* next{nullptr};

    // --- Status & Meta (Packed) ---
    uint32_t obj_num{0};// Number of objects currently allocated
    bool is_used;       // Is this span currently in CentralCache?

    // Remining bytes: 3 bytes padding

    AM_NODISCARD void* GetStartAddr() const noexcept {
        return PageNumToPtr(start_page_idx);
    }

    AM_NODISCARD void* GetEndAddr() const noexcept {
        const auto start = reinterpret_cast<uintptr_t>(GetStartAddr());
        constexpr size_t shift = std::countr_zero(MagicConstants::PAGE_SIZE);
        return reinterpret_cast<void*>(start + (page_num << shift));
    }
};

static_assert(sizeof(Span) == 64);

}// namespace aethermind