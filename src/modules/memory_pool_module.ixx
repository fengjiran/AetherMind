//
// Created by richard on 1/26/26.
//
module;

#include "macros.h"
#include "utils/logging.h"

#include <array>
#include <atomic>
#include <immintrin.h>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <vector>

export module MemoryPool;

namespace aethermind {

struct MagicConstants {
    // page size (default: 4KB)
    constexpr static size_t PAGE_SIZE = 4096;
    // page shift
    constexpr static size_t PAGE_SHIFT = 12;
    // huge page size(2MB)
    constexpr static size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
    // max thread cache size(32KB)
    constexpr static size_t MAX_TC_SIZE = 32 * 1024;
    // size class alignment
    constexpr static size_t ALIGNMENT = 16;
    // cache line size
    constexpr static size_t CACHE_LINE_SIZE = 64;

    // Maximum number of consecutive pages managed by Page Cache
    // (to avoid excessively large Spans)
    constexpr static size_t MAX_PAGE_NUM = 128;

    // bitmap bits
    constexpr static size_t BITMAP_BITS = 64;

    //
    constexpr static size_t RADIX_BITS = 9;
    constexpr static size_t RADIX_NODE_SIZE = 1 << RADIX_BITS;
    constexpr static size_t RADIX_MASK = RADIX_NODE_SIZE - 1;

    // For size class index
    constexpr static int kStepsPerGroup = 4;
    constexpr static int kStepShift = 2;
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

AM_NODISCARD constexpr void* PageIDToPtr(size_t page_idx) noexcept {
    if constexpr (std::has_single_bit(MagicConstants::PAGE_SIZE)) {
        constexpr size_t shift = std::countr_zero(MagicConstants::PAGE_SIZE);
        return reinterpret_cast<void*>(page_idx << shift);
    } else {
        return reinterpret_cast<void*>(page_idx * MagicConstants::PAGE_SIZE);
    }
}

inline void CPUPause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    // x86 环境：_mm_pause 是最稳妥的，由编译器映射为 PAUSE 指令
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM 环境：使用 ISB 或 YIELD
    __asm__ volatile("yield" ::: "memory");
#else
    // 其他架构：简单的空操作，防止编译器把循环优化掉
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

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
    static constexpr size_t Index(size_t size) noexcept {
        // Validate boundaries: size 0 or exceeding MAX_TC_SIZE are rare edge cases.
        if (size == 0 || size > MagicConstants::MAX_TC_SIZE) AM_UNLIKELY {
                return std::numeric_limits<size_t>::max();
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
        size_t base_idx = 16 + (group_idx << MagicConstants::kStepShift);
        int shift = msb - MagicConstants::kStepShift;
        size_t group_offset = ((size - 1) >> shift) & (MagicConstants::kStepsPerGroup - 1);

        return base_idx + group_offset;
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
        // Fast path for small objects (0-128 bytes): Maps index 0..15 back to 8..128
        if (idx < 16) AM_LIKELY {
                return (idx + 1) << 3;
            }

        // Decoding logarithmic stepped index
        size_t relative_idx = idx - 16;
        // Identify the binary group (2^7, 2^8, ...) and the step within it
        size_t group_idx = relative_idx >> MagicConstants::kStepShift;
        size_t step_idx = relative_idx & (MagicConstants::kStepsPerGroup - 1);
        // Reconstruct size components using 64-bit safe shifts
        size_t msb = group_idx + 7;
        size_t base_size = 1ULL << msb;
        size_t step_size = 1ULL << (msb - MagicConstants::kStepShift);
        // Return the upper bound of the current size class ladder
        return base_size + (step_idx + 1) * step_size;
    }

    /**
     * @brief Rounds up the requested size to the nearest aligned size class.
     * @param size User requested size.
     * @return Aligned size.
     */
    static constexpr size_t RoundUp(size_t size) noexcept {
        if (size > MagicConstants::MAX_TC_SIZE) AM_UNLIKELY {
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
        size_t batch = MagicConstants::MAX_TC_SIZE / size;
        // Lower bound: Always move at least 2 objects to leverage cache locality.
        if (batch < 2) {
            batch = 2;
        }

        // At most 512, to prevent the central pool from being drained instantly
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

        size_t page_num = (total_bytes + MagicConstants::PAGE_SIZE - 1) >> MagicConstants::PAGE_SHIFT;
        // 4. Boundary Enforcement
        if (page_num < 1) {
            page_num = 1;
        }

        if (page_num > MagicConstants::MAX_PAGE_NUM) {
            page_num = MagicConstants::MAX_PAGE_NUM;
        }

        return page_num;
    }
};

static_assert(SizeClass::Size(0) == 8);
static_assert(SizeClass::Size(15) == 128);
static_assert(SizeClass::Size(16) == 160);
static_assert(SizeClass::Size(19) == 256);
static_assert(SizeClass::Size(20) == 320);
// Round-trip check
static_assert(SizeClass::Index(SizeClass::Size(20)) == 20);
static_assert(SizeClass::Index(129) == 16);

struct FreeBlock {
    FreeBlock* next;
};

class FreeList {
public:
    constexpr FreeList() noexcept : head_(nullptr), size_(0) {}

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

    AM_NODISCARD void* pop() noexcept {
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
 * @brief Thread-safe, lock-free bitmap metadata for memory block management.
 *
 * This structure manages the allocation state of up to 64 memory blocks within a span.
 * Optimized for high-core-count processors by utilizing atomic CAS loops.
 */
struct Bitmap {
    /**
     * @brief Availability bitmask.
     * bit = 1: Free (Available), bit = 0: Used (Allocated).
     * Initialized to ~0ULL (all 1s) representing all blocks are free.
     */
    std::atomic<uint64_t> bits_;
    /**
     * @brief Pointer to the next bitmap node in the linked list.
     * Use atomic if the list structure itself changes concurrently (e.g., dynamic expansion).
     * If the list is fixed size at Span initialization, raw pointer is fine.
     */
    // Bitmap* next_;

    Bitmap() : bits_(~0ULL) {}

    /**
     * @brief Marks a block as free (sets bit to 1).
     * @param idx The index of the block [0-63].
     */
    void SetFree(size_t idx) noexcept {
        AM_DCHECK(idx < MagicConstants::BITMAP_BITS);
        // memory_order_release: Ensures any writes to the memory block happening
        // before this free are visible to the thread that next allocates it.
        bits_.fetch_or(1ULL << idx, std::memory_order_release);
    }

    /**
     * @brief Marks a block as used (sets bit to 0).
     * Note: Usually used during initialization or specific reservation scenarios.
     * Normal allocation should use FindAndTakeFirstFree.
     */
    void SetUsed(size_t idx) noexcept {
        AM_DCHECK(idx < MagicConstants::BITMAP_BITS);
        bits_.fetch_and(~(1ULL << idx), std::memory_order_relaxed);
    }

    /**
     * @brief Checks if a specific block is free.
     * @return True if the block is free, false otherwise.
     */
    AM_NODISCARD bool IsFree(size_t idx) const noexcept {
        AM_DCHECK(idx < MagicConstants::BITMAP_BITS);
        return (bits_.load(std::memory_order_relaxed) & (1ULL << idx)) != 0;
    }

    /**
     * @brief Check if the entire bitmap is full (all bits 0).
     * Useful for fast-path checks before attempting CAS.
     */
    AM_NODISCARD bool IsFull() const noexcept {
        return bits_.load(std::memory_order_relaxed) == 0;
    }

    /**
     * @brief Atomically finds and acquires (takes) the first available free block.
     *
     * Uses a CAS (Compare-And-Swap) loop to ensure the "find and set" operation
     * is atomic and thread-safe without using mutexes.
     *
     * @return The index of the acquired block, or MagicConstants::BITMAP_BITS if full.
     */
    AM_NODISCARD size_t FindAndTakeFirstFree() noexcept {
        // Early exit if the bitmap is full (all bits are 0).
        uint64_t old_val = bits_.load(std::memory_order_relaxed);
        do {
            if (old_val == 0) AM_UNLIKELY {
                    return MagicConstants::BITMAP_BITS;
                }
            // Calculate index of the first trailing 1 (free block).
            size_t idx = std::countr_zero(old_val);
            // Prepare the new value by flipping the bit to 0 (occupying the block).
            // Attempt to update the bitmap atomically.
            if (uint64_t new_val = old_val & ~(1ULL << idx);
                bits_.compare_exchange_weak(old_val, new_val,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) AM_LIKELY {
                    return idx;
                }
            // High-contention optimization: hint CPU to yield execution resources.
            CPUPause();
        } while (true);
    }
};

/**
 * @brief Span represents a contiguous range of memory pages.
 * Optimized for 64-bit architectures to minimize padding.
 * Total size: 64 bytes(1 cache line) to prevent false sharing and optimize fetch
 */
struct Span {
    // --- Page Cache Info ---
    size_t start_page_idx{0};// Global start page index
    size_t page_num{0};      // Number of contiguous pages

    // --- Double linked list ---
    Span* prev{nullptr};
    Span* next{nullptr};

    // --- Central Cache Object Info ---
    size_t obj_size{0};// Size of objects allocated from this Span(if applicable)
    // size_t use_count{0};
    std::atomic<size_t> use_count{0};
    size_t capacity{0};// Object capacity
    void* data_base_ptr{nullptr};

    // --- bitmap info ---
    std::atomic<uint64_t>* bitmap{nullptr};
    size_t bitmap_num{0};
    std::atomic<size_t> scan_cursor{0};

    // --- Status & Meta (Packed) ---
    bool is_used{false};// Is this span currently in CentralCache?

    void Init(size_t object_size) {
        obj_size = object_size;

        // 1. Calculate Base Address
        void* start_ptr = PageIDToPtr(start_page_idx);
        const size_t total_bytes = page_num << MagicConstants::PAGE_SHIFT;

        // 2. Estimate Bitmap Size
        // Formula: Total = BitmapBytes + DataBytes
        //          Total = (Num * 1/8) + (Num * ObjSize)
        size_t max_objs = (total_bytes * 8) / (obj_size * 8 + 1);
        bitmap_num = (max_objs + 63) / 64;
        // Placement New: Create atomic array at page start
        bitmap = new (start_ptr) std::atomic<uint64_t>[bitmap_num];

        // 3. Calculate Data Start Address (Aligned)
        uintptr_t data_start = reinterpret_cast<uintptr_t>(bitmap) + bitmap_num * 8;
        data_start = (data_start + 16 - 1) & ~(16 - 1);
        data_base_ptr = reinterpret_cast<void*>(data_start);

        // 4. Calculate Actual Capacity
        uintptr_t data_end = reinterpret_cast<uintptr_t>(start_ptr) + total_bytes;
        if (data_start >= data_end) {
            capacity = 0;
        } else {
            capacity = (data_end - data_start) / obj_size;
        }

        // 5. Initialize Bitmap Bits (Loop unrolling for performance)
        size_t full_bitmap_num = capacity / 64;
        size_t tail_bits = capacity & 63;

        // Part A: Set full blocks to ~0ULL(all free)
        for (size_t i = 0; i < full_bitmap_num; ++i) {
            bitmap[i].store(~0ULL, std::memory_order_relaxed);
        }

        // Part B: Handle the tail block(if any)
        if (full_bitmap_num < bitmap_num) {
            if (tail_bits == 0) {
                // If capacity is exact multiple of 64, this block is actually out of bounds
                // But if num_full_blocks == bitmap_num, we won't enter this branch unless i < bitmap_num
                // So this handles the case where capacity < bitmap_num * 64
                bitmap[full_bitmap_num].store(0, std::memory_order_relaxed);
            } else {
                // Set lower 'tail_bits' to 1, rest to 0
                uint64_t mask = (1ULL << tail_bits) - 1;
                // Relaxed is sufficient during initialization
                bitmap[full_bitmap_num].store(mask, std::memory_order_relaxed);
            }

            // Part C: Zero out remaining padding blocks
            for (size_t i = full_bitmap_num + 1; i < bitmap_num; ++i) {
                // Out of capacity range blocks (padding space)
                bitmap[i].store(0, std::memory_order_relaxed);
            }
        }

        use_count.store(0, std::memory_order_relaxed);
        scan_cursor.store(0, std::memory_order_relaxed);
    }

    // allocate an object
    void* AllocObject() {
        if (use_count.load(std::memory_order_relaxed) >= capacity) {
            return nullptr;
        }

        size_t idx = scan_cursor.load(std::memory_order_relaxed);
        for (size_t i = 0; i < bitmap_num; ++i) {
            size_t cur_idx = idx + i;
            if (cur_idx >= bitmap_num) {
                cur_idx -= bitmap_num;
            }

            uint64_t val = bitmap[cur_idx].load(std::memory_order_relaxed);
            if (val == 0) {
                continue;
            }

            // CAS
            while (val != 0) {
                int bit_pos = std::countr_zero(val);
                uint64_t mask = 1ULL << bit_pos;
                if (bitmap[cur_idx].compare_exchange_weak(val, val & ~mask,
                                                          std::memory_order_acquire,
                                                          std::memory_order_relaxed)) {
                    use_count.fetch_add(1, std::memory_order_relaxed);
                    if (cur_idx != idx) {
                        scan_cursor.store(cur_idx, std::memory_order_relaxed);
                    }
                    size_t global_obj_idx = cur_idx * 64 + bit_pos;
                    return static_cast<char*>(data_base_ptr) + global_obj_idx * obj_size;
                }
                CPUPause();
            }
        }
        return nullptr;
    }

    /**
     * @brief Release an object back to this Span.
     * @note Can be called concurrently without locks.
     */
    void FreeObject(void* ptr) {
        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(data_base_ptr);
        size_t global_obj_idx = offset / obj_size;

        size_t bitmap_idx = global_obj_idx / 64;
        int bit_pos = global_obj_idx & (64 - 1);
        // Release: Ensures all my writes to the object are visible
        // before the bit is marked as free.
        bitmap[bitmap_idx].fetch_or(1ULL << bit_pos, std::memory_order_release);
        // Decrement use count.
        // Release is needed if the logic checks use_count == 0 to return Span to PageCache.
        // It ensures all memory accesses in this Span are done before Span destruction/moving.
        use_count.fetch_sub(1, std::memory_order_release);
    }

    AM_NODISCARD void* GetStartAddr() const noexcept {
        return PageIDToPtr(start_page_idx);
    }

    AM_NODISCARD void* GetEndAddr() const noexcept {
        const auto start = reinterpret_cast<uintptr_t>(GetStartAddr());
        constexpr size_t shift = std::countr_zero(MagicConstants::PAGE_SIZE);
        return reinterpret_cast<void*>(start + (page_num << shift));
    }
};
// static_assert(sizeof(Span) == 64);

/**
 * @brief A doubly linked list managing a collection of Spans.
 *
 * Design Highlights:
 * 1. **Sentinel Node**: Uses a circular structure with a dummy `head_` node. This simplifies
 *    boundary checks (no nullptr checks needed for insertion/removal).
 * 2. **Bucket Locking**: Contains a mutex for fine-grained locking (typically used in CentralCache).
 * 3. **External Locking**: Core operations (insert/erase) do NOT lock internally.
 *    The caller must use GetMutex() to protect critical sections.
 */
class SpanList {
public:
    /// @brief Initializes an empty circular doubly linked list.
    /// The sentinel node's next and prev pointers point to itself.
    SpanList() noexcept {
        head_.next = &head_;
        head_.prev = &head_;
    }

    // Disable copy/move to prevent lock state corruption and pointer invalidation.
    // SpanList(const SpanList&) = delete;
    // SpanList& operator=(const SpanList&) = delete;

    /// @brief Returns a pointer to the first valid Span in the list.
    AM_NODISCARD Span* begin() noexcept {
        return head_.next;
    }

    /// @brief Returns a pointer to the sentinel node (representing the end).
    AM_NODISCARD Span* end() noexcept {
        return &head_;
    }

    /// @brief Checks if the list is empty.
    AM_NODISCARD bool empty() const noexcept {
        return head_.next == &head_;
    }

    /**
     * @brief Inserts a new Span before the specified position.
     *
     * @note Defined as `static` to avoid the overhead of passing the `this` pointer,
     * reducing register pressure in hot paths.
     *
     * @warning **Thread Safety**: The caller MUST hold the lock associated with this list.
     *
     * @param pos The position before which the new Span will be inserted.
     * @param new_span The Span to insert.
     */
    static void insert(Span* pos, Span* new_span) noexcept {
        AM_DCHECK(pos != nullptr && new_span != nullptr);
        new_span->next = pos;
        new_span->prev = pos->prev;
        new_span->prev->next = new_span;
        pos->prev = new_span;
    }

    /**
    * @brief Inserts a Span at the beginning of the list.
    *
    * This is typically used when returning memory to the cache.
    * LIFO (Last-In-First-Out) behavior improves CPU cache locality for hot data.
    */
    void push_front(Span* span) noexcept {
        insert(begin(), span);
    }

    /**
    * @brief Inserts a Span at the end of the list.
    *
    * This is typically used when returning memory to the cache.
    * LIFO (Last-In-First-Out) behavior improves CPU cache locality for hot data.
    */
    void push_back(Span* span) noexcept {
        insert(end(), span);
    }

    /**
     * @brief Unlinks a Span from the list.
     *
     * @note This function only detaches the node; it does NOT `delete` the memory.
     * The Span's lifecycle is managed by the PageCache.
     *
     * @param pos The Span to remove. Must not be `nullptr` or the sentinel `head_`.
     * @return Span* Pointer to the next node (useful for iteration).
     */
    Span* erase(Span* pos) noexcept {
        AM_DCHECK(pos != nullptr && pos != &head_);
        auto* prev = pos->prev;
        auto* next = pos->next;
        prev->next = next;
        next->prev = prev;
        pos->prev = nullptr;
        pos->next = nullptr;
        return next;
    }

    /**
     * @brief Removes and returns the first Span in the list.
     * @return Span* Pointer to the popped Span, or `nullptr` if the list is empty.
     */
    Span* pop_front() noexcept {
        // Hint to the compiler that an empty list is unlikely in the hot path.
        if (empty()) AM_UNLIKELY {
                return nullptr;
            }

        auto* pos = head_.next;
        erase(pos);
        return pos;
    }

    /// @brief Accessor for the bucket lock.
    /// @return Reference to the mutex, intended for `std::lock_guard`.
    AM_NODISCARD std::mutex& GetMutex() noexcept {
        return mutex_;
    }

private:
    /**
    * @brief Sentinel node (Dummy Head).
    * Stored as a member object (not a pointer) to ensure it resides in the
    * same cache line as the SpanList object itself.
    */
    Span head_;
    /// @brief Mutex protecting concurrent access to this specific list (Bucket Lock).
    std::mutex mutex_;
};

/**
 * @brief Node structure for the Radix Tree (PageMap).
 *
 * Maps Page IDs (keys) to Span pointers (values).
 *
 * @note **Alignment**: `alignas(PAGE_SIZE)` forces the structure to be 4KB aligned.
 * 1. Ensures that one node occupies exactly one physical OS page (assuming 4KB pages).
 * 2. Prevents False Sharing in multi-thread environments.
 * 3. Optimizes interaction with system allocators (like mmap).
 */
struct alignas(MagicConstants::PAGE_SIZE) RadixNode {
    /**
     * @brief Array of pointers to child nodes or Spans.
     *
     * - Size is typically 512 for 64-bit systems (9 bits stride).
     * - In leaf nodes, these point to `Span` objects.
     * - In internal nodes, these point to the next level `RadixNode`.
     */
    std::array<std::atomic<void*>, MagicConstants::RADIX_NODE_SIZE> children;

    RadixNode() {
        for (auto& child: children) {
            child.store(nullptr, std::memory_order_relaxed);
        }
    }
};

class PageMap {
public:
    /**
     * @brief Lookup the Span associated with a specific memory address.
     *
     * This function is lock-free and extremely hot in the deallocation path.
     * It relies on the memory barriers established by SetSpan to ensure data visibility.
     *
     * @param page_id The page id being freed or looked up.
     * @return Span* Pointer to the managing Span, or nullptr if not found.
     */
    static Span* GetSpan(size_t page_id) {
        // Acquire semantics ensure we see the initialized data of the root node
        // if it was just created by another thread.
        auto* curr = root_.load(std::memory_order_acquire);
        if (!curr) AM_UNLIKELY {
                return nullptr;
            }

        // Calculate Radix Tree indices
        const size_t i1 = page_id >> (MagicConstants::RADIX_BITS * 2);
        const size_t i2 = (page_id >> MagicConstants::RADIX_BITS) & MagicConstants::RADIX_MASK;
        const size_t i3 = page_id & MagicConstants::RADIX_MASK;

        // Traverse Level 1
        auto* p2_raw = curr->children[i1].load(std::memory_order_acquire);
        if (!p2_raw) AM_UNLIKELY {
                return nullptr;
            }
        auto* p2 = static_cast<RadixNode*>(p2_raw);

        // Traverse Level 2
        auto* p3_raw = p2->children[i2].load(std::memory_order_acquire);
        if (!p3_raw) AM_UNLIKELY {
                return nullptr;
            }
        auto* p3 = static_cast<RadixNode*>(p3_raw);

        // Fetch Level 3 (Leaf)
        // Note: On weak memory models (ARM), an acquire fence might be technically required here
        // to ensure the content of the returned Span is visible. However, on x86-64,
        // data dependency usually suffices.
        return static_cast<Span*>(p3->children[i3].load(std::memory_order_acquire));
    }

    static Span* GetSpan(void* ptr) {
        const auto addr = reinterpret_cast<uintptr_t>(ptr);
        const size_t page_id = addr >> MagicConstants::PAGE_SHIFT;
        return GetSpan(page_id);
    }

    /**
    * @brief Register a Span into the PageMap.
    *
    * Associates all page IDs covered by the span with the span pointer.
    * This operation holds a lock to protect the tree structure during growth.
    *
    * @param span The Span to register. Must have valid start_page_idx and page_num.
    */
    static void SetSpan(Span* span) {
        // Lock protects the tree structure from concurrent modifications.
        std::lock_guard<std::mutex> lock(mutex_);
        auto* curr = root_.load(std::memory_order_relaxed);
        if (!curr) {
            curr = new RadixNode;
            root_.store(curr, std::memory_order_release);
        }

        const auto start = span->start_page_idx;
        const auto page_num = span->page_num;

        for (size_t i = 0; i < page_num; ++i) {
            uintptr_t page_id = start + i;
            // 1. Ensure the path (Intermediate Nodes) exists.
            // Note: Called WITHOUT internal locking to avoid deadlock.
            auto* p3 = EnsurePath(curr, page_id);
            // 2. Traverse to the leaf node.
            const size_t i3 = page_id & MagicConstants::RADIX_MASK;
            // 3. Set the Leaf.
            p3->children[i3].store(span, std::memory_order_release);
        }
    }

private:
    // Atomic root pointer for double-checked locking / lazy initialization.
    inline static std::atomic<RadixNode*> root_ = nullptr;
    // Mutex protects tree growth (new node allocation).
    inline static std::mutex mutex_;

    /**
     * @brief Helper to create missing intermediate nodes for a given Page ID.
     *
     * @warning **MUST be called with 'mutex_' held.**
     * Internal helper function. Does not lock internally to prevent recursive deadlock.
     */
    static RadixNode* EnsurePath(RadixNode* curr, uintptr_t page_id) {
        // Step 1: Ensure Level 2 Node exists
        const size_t i1 = page_id >> (MagicConstants::RADIX_BITS * 2);
        const size_t i2 = (page_id >> MagicConstants::RADIX_BITS) & MagicConstants::RADIX_MASK;
        auto* p2_raw = curr->children[i1].load(std::memory_order_relaxed);
        if (!p2_raw) {
            auto* new_node = new RadixNode;
            curr->children[i1].store(new_node, std::memory_order_release);
            p2_raw = new_node;
        }

        // Step 2: Ensure Level 3 Node exists
        auto* p2 = static_cast<RadixNode*>(p2_raw);
        auto* p3_raw = p2->children[i2].load(std::memory_order_relaxed);
        if (!p3_raw) {
            auto* new_node = new RadixNode;
            p2->children[i2].store(new_node, std::memory_order_release);
            p3_raw = new_node;
        }
        return static_cast<RadixNode*>(p3_raw);
    }
};

class PageAllocator {
public:
    static void* Allocate(size_t page_num) {
        const size_t size = page_num << MagicConstants::PAGE_SHIFT;
        if (size < (MagicConstants::HUGE_PAGE_SIZE >> 1)) AM_LIKELY {
                return AllocNormalPage(size);
            }

        return AllocHugePage(size);
    }

    static void Release(void* ptr, size_t page_num) {
        if (!ptr || page_num == 0) {
            return;
        }

        const size_t size = page_num << MagicConstants::PAGE_SHIFT;
        munmap(ptr, size);
    }

private:
    static void* AllocNormalPage(size_t size) {
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        return ptr;
    }

    static void* AllocHugePage(size_t size) {
        size_t alloc_size = size + MagicConstants::HUGE_PAGE_SIZE;
        void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        const auto addr = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t aligned_addr = (addr + MagicConstants::HUGE_PAGE_SIZE - 1) &
                                       ~(MagicConstants::HUGE_PAGE_SIZE - 1);
        const size_t head_gap = aligned_addr - addr;
        if (head_gap > 0) {
            munmap(ptr, head_gap);
        }

        if (const size_t tail_gap = alloc_size - head_gap - size; tail_gap > 0) {
            munmap(reinterpret_cast<void*>(aligned_addr + size), tail_gap);
        }

        madvise(reinterpret_cast<void*>(aligned_addr), size, MADV_HUGEPAGE);
        return reinterpret_cast<void*>(aligned_addr);
    }
};

/**
 * @brief Global singleton managing page-level memory allocation and deallocation.
 *
 * The PageCache is the central repository for Spans (contiguous memory pages).
 * It sits above the OS memory allocator (PageAllocator) and below the CentralCache.
 *
 * Key Responsibilities:
 * 1. **Distribution**: Slices large spans into smaller ones for CentralCache.
 * 2. **Coalescing**: Merges adjacent free spans returned by CentralCache to reduce external fragmentation.
 * 3. **System Interaction**: Requests large memory blocks from the OS when the cache is empty.
 */
class PageCache {
public:
    /**
    * @brief Retrieves the singleton instance of PageCache.
    */
    static PageCache& GetInstance() {
        static PageCache instance;
        return instance;
    }

    // Disable copy and assignment to enforce singleton pattern.
    PageCache(const PageCache&) = delete;
    PageCache& operator=(const PageCache&) = delete;

    /**
     * @brief Allocates a Span with at least `page_num` pages.
     *
     * Thread-safe wrapper that acquires the global lock.
     *
     * @param page_num Number of pages requested.
     * @param obj_size Size of the objects this Span will manage (metadata for CentralCache).
     * @return Pointer to the allocated Span.
     */
    Span* AllocSpan(size_t page_num, size_t obj_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        return AllocSpanLocked(page_num, obj_size);
    }

    /**
     * @brief Returns a Span to the PageCache and attempts to merge it with neighbors.
     *
     * This function performs **Physical Coalescing**:
     * 1. Checks left and right neighbors using the PageMap.
     * 2. If neighbors are free and the total size is within limits, merges them.
     * 3. Inserts the resulting (potentially larger) Span back into the free list.
     *
     * @param span The Span to be released.
     */
    void ReleaseSpan(Span* span) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. Direct Return: If the Span is larger than the cache can manage (>128 pages),
        // return it directly to the OS (PageAllocator).
        if (span->page_num > MagicConstants::MAX_PAGE_NUM) AM_UNLIKELY {
                auto* ptr = span->GetStartAddr();
                PageAllocator::Release(ptr, span->page_num);
                delete span;
                return;
            }

        // 2. Merge Left: Check the previous page ID.
        while (true) {
            size_t left_id = span->start_page_idx - 1;
            // Retrieve the Span managing the left page from the global PageMap.
            auto* left_span = PageMap::GetSpan(left_id);
            // Stop merging if:
            // - Left page doesn't exist (not managed by us).
            // - Left span is currently in use (in CentralCache).
            // - Merged size would exceed the maximum bucket size.
            if (!left_span || left_span->is_used ||
                span->page_num + left_span->page_num > MagicConstants::MAX_PAGE_NUM) {
                break;
            }

            // Perform merge: Remove left_span from its list, absorb it into 'span'.
            span_lists_[left_span->page_num].erase(left_span);
            span->start_page_idx = left_span->start_page_idx;// Update start to left
            span->page_num += left_span->page_num;           // Increase size
            delete left_span;                                // Destroy metadata
        }

        // 3. Merge Right: Check the page ID immediately following this span.
        while (true) {
            size_t right_id = span->start_page_idx + span->page_num;
            auto* right_span = PageMap::GetSpan(right_id);
            // Similar stop conditions as Merge Left.
            if (!right_span || right_span->is_used ||
                span->page_num + right_span->page_num > MagicConstants::MAX_PAGE_NUM) {
                break;
            }

            // Perform merge: Remove right_span, absorb it.
            span_lists_[right_span->page_num].erase(right_span);
            span->page_num += right_span->page_num;// Start index stays same, size increases
            delete right_span;
        }

        // 4. Insert back: Mark as unused and push to the appropriate bucket.
        span->is_used = false;
        span->obj_size = 0;
        span_lists_[span->page_num].push_front(span);
        // Update PageMap: Map ALL pages in this coalesced span to the span pointer.
        // This ensures subsequent merge operations can find this span via any of its pages.
        PageMap::SetSpan(span);
    }

    AM_NODISCARD std::mutex& GetMutex() noexcept {
        return mutex_;
    }

private:
    /// Global lock protecting the span_lists_ structure.
    std::mutex mutex_;
    /// Array of free lists. Index `i` holds Spans of size `i` pages.
    /// Range: [0, MAX_PAGE_NUM], supporting spans up to 128 pages.
    std::array<SpanList, MagicConstants::MAX_PAGE_NUM + 1> span_lists_;

    PageCache() = default;

    /**
     * @brief Internal core logic for allocation (assumes lock is held).
     * Uses a loop to handle system refill and splitting.
     */
    Span* AllocSpanLocked(size_t page_num, size_t obj_size) {
        while (true) {
            // 1. Oversized Allocation:
            // Requests larger than the max bucket (>128 pages) go directly to the OS.
            if (page_num > MagicConstants::MAX_PAGE_NUM) AM_UNLIKELY {
                    void* ptr = PageAllocator::Allocate(page_num);
                    auto* span = new Span;
                    span->start_page_idx = reinterpret_cast<uintptr_t>(ptr) >> MagicConstants::PAGE_SHIFT;
                    span->page_num = page_num;
                    span->obj_size = obj_size;
                    span->is_used = true;

                    // Register relationship in Radix Tree.
                    PageMap::SetSpan(span);
                    return span;
                }

            // 2. Exact Match:
            // Check if there is a free span in the bucket corresponding exactly to page_num.
            if (!span_lists_[page_num].empty()) {
                auto* span = span_lists_[page_num].pop_front();
                span->obj_size = obj_size;
                span->is_used = true;
                return span;
            }

            // 3. Splitting (Best Fit / First Fit):
            // Iterate through larger buckets to find a span we can split.
            for (size_t i = page_num + 1; i <= MagicConstants::MAX_PAGE_NUM; ++i) {
                if (span_lists_[i].empty()) {
                    continue;
                }

                // Found a larger span.
                auto* big_span = span_lists_[i].pop_front();
                // Create a new span for the requested `page_num` (Head Split).
                auto* small_span = new Span;
                small_span->start_page_idx = big_span->start_page_idx;
                small_span->page_num = page_num;
                small_span->obj_size = obj_size;
                small_span->is_used = true;

                // Adjust the remaining part of the big span (Tail).
                big_span->start_page_idx += page_num;
                big_span->page_num -= page_num;
                big_span->is_used = false;
                // Return the remainder to the appropriate free list.
                span_lists_[big_span->page_num].push_front(big_span);

                // Register both parts in the PageMap.
                PageMap::SetSpan(small_span);
                PageMap::SetSpan(big_span);
                return small_span;
            }

            // 4. System Refill:
            // If no suitable spans exist in cache, allocate a large block (128 pages) from OS.
            // We request the MAX_PAGE_NUM to maximize cache efficiency.
            size_t alloc_page_nums = MagicConstants::MAX_PAGE_NUM;
            void* ptr = PageAllocator::Allocate(alloc_page_nums);
            auto* span = new Span;
            span->start_page_idx = reinterpret_cast<uintptr_t>(ptr) >> MagicConstants::PAGE_SHIFT;
            span->page_num = alloc_page_nums;
            span->is_used = false;
            // Insert the new large span into the last bucket.
            span_lists_[alloc_page_nums].push_front(span);
            PageMap::SetSpan(span);
            // Continue the loop:
            // The next iteration will jump to step 3 (Splitting), finding the
            // 128-page span we just added, splitting it, and returning the result.
        }
    }
};

/**
 * @brief Central resource manager connecting ThreadCache and PageCache.
 *
 * CentralCache acts as a hub that balances memory resources among multiple threads.
 * It divides memory into different "Size Classes" (Buckets), each protected by a separate lock (Bucket Lock).
 *
 * Key Responsibilities:
 * 1. **Distribution**: Fetches large Spans from PageCache, slices them into objects, and serves ThreadCache in batches.
 * 2. **Recycling**: Receives returned objects from ThreadCache and releases Spans back to PageCache when they are completely empty.
 * 3. **Concurrency**: Reduces lock contention using fine-grained bucket locks compared to the single global lock in PageCache.
 */
class CentralCache {
public:
    /**
     * @brief Singleton Accessor.
     */
    static CentralCache& GetInstance() {
        static CentralCache instance;
        return instance;
    }

    // Disable copy/move to enforce singleton pattern.
    CentralCache(const CentralCache&) = delete;
    CentralCache& operator=(const CentralCache&) = delete;

    /**
     * @brief Fetches a batch of objects for a specific ThreadCache.
     *
     * This function pulls objects from the non-empty spans in the corresponding bucket.
     * If the bucket is empty or exhausted, it requests a new Span from PageCache.
     *
     * @param block_list Output parameter. The fetched objects are pushed into this FreeList.
     * @param batch_num The desired number of objects to fetch.
     * @param size The size of the object (used to determine the bucket index).
     * @return size_t The actual number of objects fetched (may be less than batch_num).
     */
    size_t FetchRange(FreeList& block_list, size_t batch_num, size_t size) {
        auto idx = SizeClass::Index(size);
        auto& span_list = span_lists_[idx];

        // Apply Bucket Lock (Fine-grained locking).
        std::unique_lock<std::mutex> lock(span_list.GetMutex());

        size_t i = 0;
        void* batch_head = nullptr;
        void* batch_tail = nullptr;
        // Try to fulfill the batch request
        while (i < batch_num) {
            // 1. Refill Logic:
            // If list is empty OR the current head span is fully allocated, get a new span.
            // Note: use_count check is a fast-path hint; AllocObject is the authority.
            if (span_list.empty() ||
                span_list.begin()->use_count.load(std::memory_order_relaxed) >= span_list.begin()->capacity) {
                // GetOneSpan releases the bucket lock internally to avoid deadlock with PageCache.
                if (!GetOneSpan(span_list, size)) {
                    break;
                }
            }

            // 2. Allocation Loop:
            // Take the first span (LRU strategy: valid spans are at front, full ones at back).
            auto* span = span_list.begin();
            while (i < batch_num) {
                void* obj = span->AllocObject();
                if (!obj) {
                    // Current span is full. Move it to the end of the list.
                    // This ensures subsequent allocations check other spans first.
                    span_list.erase(span);
                    span_list.push_back(span);
                    break;// Break inner loop to check the next span or refill.
                }

                // 3. Link objects into a temporary list (LIFO / Head Insert)
                auto* node = static_cast<FreeBlock*>(obj);
                if (batch_head == nullptr) {
                    batch_tail = obj;// First node allocated is the tail of the batch.
                }

                node->next = static_cast<FreeBlock*>(batch_head);
                batch_head = node;
                ++i;
            }
        }

        lock.unlock();
        // 4. Batch Push: Move the collected objects to ThreadCache's FreeList.
        if (i > 0) {
            block_list.push_range(batch_head, batch_tail, i);
        }
        return i;
    }

    /**
    * @brief returns a batch of objects from ThreadCache to CentralCache.
    *
    * Iterates through the list, finds the owning Span for each object via PageMap,
    * and releases the object. May trigger Span release to PageCache.
    *
    * @param start Head of the linked list of objects to release.
    * @param size Size of the objects (must match the bucket).
    */
    void ReleaseListToSpans(void* start, size_t size) {
        auto idx = SizeClass::Index(size);
        auto& span_list = span_lists_[idx];

        std::unique_lock<std::mutex> lock(span_list.GetMutex());
        while (start) {
            void* next = static_cast<FreeBlock*>(start)->next;
            // 1. Identify the Span owning this object.
            auto* span = PageMap::GetSpan(start);
            AM_DCHECK(span != nullptr);
            AM_DCHECK(span->obj_size == size);
            // 2. Return object to Span.
            span->FreeObject(start);

            // 3. Heuristic: If a full span becomes non-full, move it to the front.
            // This allows FetchRange to immediately find this available slot.
            if (span->use_count.load(std::memory_order_relaxed) == span->capacity - 1) {
                span_list.erase(span);
                span_list.push_front(span);
            }

            // 4. Release to PageCache:
            // If the span becomes completely empty, return it to PageCache for coalescing.
            if (span->use_count.load(std::memory_order_relaxed) == 0) {
                span_list.erase(span);
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

            start = next;
        }
    }

private:
    CentralCache() = default;

    /**
     * @brief Refills the SpanList by requesting a new Span from PageCache.
     * @warning Must be called with the bucket lock HELD. Will temporarily release it.
     */
    static Span* GetOneSpan(SpanList& list, size_t size) {
        // 1. Unlock bucket lock to perform expensive PageCache operation.
        list.GetMutex().unlock();

        // 2. Calculate optimal page count and request from PageCache.
        // Assuming SizeClass::GetMovePageNum is equivalent to NumMovePage discussed earlier.
        auto page_num = SizeClass::GetMovePageNum(size);
        auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
        AM_DCHECK(span != nullptr);
        // 3. Initialize the new Span (Slice it into objects).
        // This accesses new memory, safe to do without lock (thread-local at this point).
        span->Init(size);// NOLINT
        // 4. Re-lock and push to the list.
        list.GetMutex().lock();
        list.push_front(span);
        return span;
    }

    constexpr static size_t kNumSizeClasses = SizeClass::Index(MagicConstants::MAX_TC_SIZE) + 1;
    std::array<SpanList, kNumSizeClasses> span_lists_{};
};

class alignas(64) ThreadCache {
public:
    ThreadCache() noexcept = default;

    ThreadCache(const ThreadCache&) = delete;
    ThreadCache& operator=(const ThreadCache&) = delete;

    AM_NODISCARD void* Allocate(size_t size) noexcept {
        if (size == 0) AM_UNLIKELY {
                return nullptr;
            }

        // large object

        // small object
        size_t sc_idx = SizeClass::Index(size);
        auto& free_list = free_lists_[sc_idx];
        auto* ptr = free_list.pop();
        if (!ptr) {// allocate from Central Cache
            //
        }
    }

private:
    constexpr static size_t kNumSizeClasses = SizeClass::Index(MagicConstants::MAX_TC_SIZE) + 1;
    std::array<FreeList, kNumSizeClasses> free_lists_{};
};

/**
 * @brief The global thread-local instance.
 * Initialized lazily upon the first access by each thread.
 */
inline ThreadCache& GetThreadCache() noexcept {
    thread_local ThreadCache inst;
    return inst;
}

}// namespace aethermind