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
#include <thread>
#include <vector>

export module MemoryPool;

namespace aethermind {

struct MagicConstants {
    // page size (default: 4KB)
    constexpr static size_t PAGE_SIZE = 4096;
    // page shift
    constexpr static size_t PAGE_SHIFT = 12;
    // max thread cache size(32KB)
    constexpr static size_t MAX_TC_SIZE = 32 * 1024;
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
AM_NODISCARD constexpr size_t GetSizeClassIndexFromSize(size_t size) noexcept {
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
    int base_idx = 16 + (group_idx << MagicConstants::kStepShift);
    int shift = msb - MagicConstants::kStepShift;
    int group_offset = ((size - 1) >> shift) & (MagicConstants::kStepsPerGroup - 1);

    return base_idx + group_offset;
}

/**
 * @brief Reconstructs the maximum object size for a given size class index.
 *
 * This function serves as the exact inverse of GetSizeClassIndexFromSize.
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
 * This ensures that $GetSizeFromSizeClassIndex(GetSizeClassIndexFromSize(s)) \ge s$
 * for any $s \in (0, MAX\_TC\_SIZE]$.
 *
 * @param idx The size class index to be decoded.
 * @return The maximum byte size of the objects stored in this size class's FreeList.
 */
AM_NODISCARD constexpr size_t GetSizeFromSizeClassIndex(size_t idx) noexcept {
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

static_assert(GetSizeFromSizeClassIndex(0) == 8);
static_assert(GetSizeFromSizeClassIndex(15) == 128);
static_assert(GetSizeFromSizeClassIndex(16) == 160);
static_assert(GetSizeFromSizeClassIndex(19) == 256);
static_assert(GetSizeFromSizeClassIndex(20) == 320);
// Round-trip check
static_assert(GetSizeClassIndexFromSize(GetSizeFromSizeClassIndex(20)) == 20);
static_assert(GetSizeClassIndexFromSize(129) == 16);

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
        return reinterpret_cast<char*>(block) + sizeof(FreeBlock);
    }

private:
    FreeBlock* head_;
    size_t size_;
};

/**
 * @brief Thread-safe, lock-free bitmap metadata for memory block management.
 *
 * This structure manages the allocation state of up to 64 memory blocks within a slab.
 * Optimized for high-core-count processors by utilizing atomic CAS loops
 * and cache-line alignment to prevent false sharing.
 */
struct alignas(64) BitmapMeta {
    /**
     * @brief Availability bitmask.
     * bit = 1: Free (Available), bit = 0: Used (Allocated).
     */
    std::atomic<uint64_t> bits_;
    /** @brief Pointer to the next bitmap node in the linked list. */
    BitmapMeta* next_;

    BitmapMeta() : bits_(~0ULL), next_(nullptr) {}

    /**
     * @brief Marks a block as free (sets bit to 1).
     * @param idx The index of the block [0-63].
     */
    void SetFree(size_t idx) noexcept {
        bits_.fetch_or(1ULL << idx, std::memory_order_release);
    }

    /**
     * @brief Marks a block as used (sets bit to 0).
     * @param idx The index of the block [0-63].
     */
    void SetUsed(size_t idx) noexcept {
        bits_.fetch_and(~(1ULL << idx), std::memory_order_relaxed);
    }

    /**
     * @brief Checks if a specific block is free.
     * @return True if the block is free, false otherwise.
     */
    AM_NODISCARD bool IsFree(size_t idx) const noexcept {
        return (bits_.load(std::memory_order_relaxed) & (1ULL << idx)) != 0;
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
    size_t obj_size{0};      // Size of objects allocated from this Span(if applicable)
    size_t obj_num{0};       // Number of objects currently allocated
    void* free_list{nullptr};// Embedded free list for small object allocation

    // --- Status & Meta (Packed) ---
    bool is_used{false};// Is this span currently in CentralCache?

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
     * - Value initialization `{}` guarantees all pointers start as `nullptr`.
     * - In leaf nodes, these point to `Span` objects.
     * - In internal nodes, these point to the next level `RadixNode`.
     */
    std::array<void*, MagicConstants::RADIX_NODE_SIZE> children{};

    AM_NODISCARD RadixNode* GetChild(size_t idx) const {
        return static_cast<RadixNode*>(children[idx]);
    }

    AM_NODISCARD Span* GetSpan(size_t idx) const {
        return static_cast<Span*>(children[idx]);
    }

    void Set(size_t idx, void* ptr) {
        children[idx] = ptr;
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
     * @param ptr The pointer to the object being freed or looked up.
     * @return Span* Pointer to the managing Span, or nullptr if not found.
     */
    static Span* GetSpan(void* ptr) {
        // Acquire semantics ensure we see the initialized data of the root node
        // if it was just created by another thread.
        auto* curr = root_.load(std::memory_order_acquire);
        if (!curr) {
            return nullptr;
        }

        // Calculate Page ID and Radix Tree indices
        const auto page_id = reinterpret_cast<uintptr_t>(ptr) >> MagicConstants::PAGE_SHIFT;
        const size_t i1 = page_id >> (MagicConstants::RADIX_BITS * 2);
        const size_t i2 = (page_id >> MagicConstants::RADIX_BITS) & MagicConstants::RADIX_MASK;
        const size_t i3 = page_id & MagicConstants::RADIX_MASK;

        // Traverse Level 1
        if (!curr->children[i1]) {
            return nullptr;
        }
        auto* p2 = static_cast<RadixNode*>(curr->children[i1]);

        // Traverse Level 2
        if (!p2->children[i2]) {
            return nullptr;
        }
        auto* p3 = static_cast<RadixNode*>(p2->children[i2]);

        // Fetch Level 3 (Leaf)
        // Note: On weak memory models (ARM), an acquire fence might be technically required here
        // to ensure the content of the returned Span is visible. However, on x86-64,
        // data dependency usually suffices.
        return static_cast<Span*>(p3->children[i3]);
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
            EnsurePath(curr, page_id);
            // 2. Traverse to the leaf node.
            const size_t i1 = page_id >> (MagicConstants::RADIX_BITS * 2);
            auto* p2 = static_cast<RadixNode*>(curr->children[i1]);
            const size_t i2 = (page_id >> MagicConstants::RADIX_BITS) & MagicConstants::RADIX_MASK;
            auto* p3 = static_cast<RadixNode*>(p2->children[i2]);
            const size_t i3 = page_id & MagicConstants::RADIX_MASK;
            // 3. Set the Leaf.
            // CRITICAL: Release fence ensures that the fully initialized 'span' object
            // is visible to any thread that reads this pointer via GetSpan.
            std::atomic_thread_fence(std::memory_order_release);
            p3->children[i3] = span;
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
    static void EnsurePath(RadixNode* curr, uintptr_t page_id) {
        // Step 1: Ensure Level 2 Node exists
        const size_t i1 = page_id >> (MagicConstants::RADIX_BITS * 2);
        const size_t i2 = (page_id >> MagicConstants::RADIX_BITS) & MagicConstants::RADIX_MASK;
        if (!curr->children[i1]) {
            auto* new_node = new RadixNode;
            // Fence prevents reordering of pointer assignment before initialization
            std::atomic_thread_fence(std::memory_order_release);
            curr->children[i1] = new_node;
        }

        // Step 2: Ensure Level 3 Node exists
        auto* n2 = static_cast<RadixNode*>(curr->children[i1]);
        if (!n2->children[i2]) {
            auto* new_node = new RadixNode;
            // Fence prevents reordering of pointer assignment before initialization
            std::atomic_thread_fence(std::memory_order_release);
            n2->children[i2] = new_node;
        }
    }
};

class PageAllocator {
public:
    static PageAllocator& GetInstance() noexcept {
        static PageAllocator instance;
        return instance;
    }

    PageAllocator(const PageAllocator&) = delete;
    PageAllocator& operator=(const PageAllocator&) = delete;

    AM_NODISCARD Span* AllocateSpan(size_t page_num) noexcept {
        if (page_num == 0 || page_num > MagicConstants::MAX_PAGE_NUM) AM_UNLIKELY {
                return nullptr;
            }
    }

private:
    PageAllocator() noexcept {
        // span_lists_.resize(MagicConstants::MAX_PAGE_NUM + 1);
        // span_lists_.resize(10);
    }

    RadixNode root_;
    std::vector<SpanList> span_lists_;
    std::mutex mutex_;
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
        size_t sc_idx = GetSizeClassIndexFromSize(size);
        auto& free_list = free_lists_[sc_idx];
        auto* ptr = free_list.pop();
        if (!ptr) {// allocate from Central Cache
            //
        }
    }

private:
    constexpr static size_t kNumSizeClasses = GetSizeClassIndexFromSize(MagicConstants::MAX_TC_SIZE) + 1;
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