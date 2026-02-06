//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_AMMALLOC_MEMORY_POOL_H
#define AETHERMIND_AMMALLOC_MEMORY_POOL_H

#include "utils/logging.h"
#include "ammalloc/config.h"
#include "ammalloc/size_class.h"

#include <atomic>
#include <immintrin.h>
#include <mutex>
#include <sys/mman.h>

namespace aethermind {

/**
 * @brief Aligns 'size' up to the specified 'align'.
 * @return The smallest multiple of 'align' that is greater than or equal to 'size'.
 */
AM_NODISCARD constexpr size_t AlignUp(size_t size,
                                      size_t align = SystemConfig::ALIGNMENT) noexcept {
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
AM_NODISCARD inline size_t PtrToPageIdx(void* ptr) noexcept {
    // 1. Cast pointer to integer. Note: This limits 'true' constexpr usage
    // but enables massive runtime inlining optimizations.
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    // 2. Static dispatch for page size alignment.
    if constexpr (std::has_single_bit(SystemConfig::PAGE_SIZE)) {
        // Optimization: Address / 2^n -> Address >> n
        constexpr size_t shift = std::countr_zero(SystemConfig::PAGE_SIZE);
        return addr >> shift;
    } else {
        // Fallback for non-standard page sizes.
        return addr / SystemConfig::PAGE_SIZE;
    }
}

AM_NODISCARD inline void* PageIDToPtr(size_t page_idx) noexcept {
    if constexpr (std::has_single_bit(SystemConfig::PAGE_SIZE)) {
        constexpr size_t shift = std::countr_zero(SystemConfig::PAGE_SIZE);
        return reinterpret_cast<void*>(page_idx << shift);
    } else {
        return reinterpret_cast<void*>(page_idx * SystemConfig::PAGE_SIZE);
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
    constexpr FreeList() noexcept : head_(nullptr), size_(0), max_size_(1) {}

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

    AM_NODISCARD size_t max_size() const noexcept {
        return max_size_;
    }

    void set_max_size(size_t n) noexcept {
        max_size_ = n;
    }

private:
    FreeBlock* head_;
    size_t size_;
    size_t max_size_;
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
        const size_t total_bytes = page_num << SystemConfig::PAGE_SHIFT;

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
        constexpr size_t shift = std::countr_zero(SystemConfig::PAGE_SIZE);
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
struct alignas(SystemConfig::PAGE_SIZE) RadixNode {
    /**
     * @brief Array of pointers to child nodes or Spans.
     *
     * - Size is typically 512 for 64-bit systems (9 bits stride).
     * - In leaf nodes, these point to `Span` objects.
     * - In internal nodes, these point to the next level `RadixNode`.
     */
    std::array<std::atomic<void*>, PageConfig::RADIX_NODE_SIZE> children;

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
        const size_t i1 = page_id >> (PageConfig::RADIX_BITS * 2);
        const size_t i2 = (page_id >> PageConfig::RADIX_BITS) & PageConfig::RADIX_MASK;
        const size_t i3 = page_id & PageConfig::RADIX_MASK;

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
        const size_t page_id = addr >> SystemConfig::PAGE_SHIFT;
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
            const size_t i3 = page_id & PageConfig::RADIX_MASK;
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
        const size_t i1 = page_id >> (PageConfig::RADIX_BITS * 2);
        const size_t i2 = (page_id >> PageConfig::RADIX_BITS) & PageConfig::RADIX_MASK;
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
    static void* SystemAlloc(size_t page_num) {
        const size_t size = page_num << SystemConfig::PAGE_SHIFT;
        if (size < (SystemConfig::HUGE_PAGE_SIZE >> 1)) AM_LIKELY {
                return AllocNormalPage(size);
            }

        return AllocHugePage(size);
    }

    static void SystemFree(void* ptr, size_t page_num) {
        if (!ptr || page_num == 0) {
            return;
        }

        const size_t size = page_num << SystemConfig::PAGE_SHIFT;
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
        size_t alloc_size = size + SystemConfig::HUGE_PAGE_SIZE;
        void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        const auto addr = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t aligned_addr = (addr + SystemConfig::HUGE_PAGE_SIZE - 1) &
                                       ~(SystemConfig::HUGE_PAGE_SIZE - 1);
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
        if (span->page_num > PageConfig::MAX_PAGE_NUM) AM_UNLIKELY {
                auto* ptr = span->GetStartAddr();
                PageAllocator::SystemFree(ptr, span->page_num);
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
                span->page_num + left_span->page_num > PageConfig::MAX_PAGE_NUM) {
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
                span->page_num + right_span->page_num > PageConfig::MAX_PAGE_NUM) {
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
    std::array<SpanList, PageConfig::MAX_PAGE_NUM + 1> span_lists_;

    PageCache() = default;

    /**
     * @brief Internal core logic for allocation (assumes lock is held).
     * Uses a loop to handle system refill and splitting.
     */
    Span* AllocSpanLocked(size_t page_num, size_t obj_size) {
        while (true) {
            // 1. Oversized Allocation:
            // Requests larger than the max bucket (>128 pages) go directly to the OS.
            if (page_num > PageConfig::MAX_PAGE_NUM) AM_UNLIKELY {
                    void* ptr = PageAllocator::SystemAlloc(page_num);
                    auto* span = new Span;
                    span->start_page_idx = reinterpret_cast<uintptr_t>(ptr) >> SystemConfig::PAGE_SHIFT;
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
            for (size_t i = page_num + 1; i <= PageConfig::MAX_PAGE_NUM; ++i) {
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
            size_t alloc_page_nums = PageConfig::MAX_PAGE_NUM;
            void* ptr = PageAllocator::SystemAlloc(alloc_page_nums);
            auto* span = new Span;
            span->start_page_idx = reinterpret_cast<uintptr_t>(ptr) >> SystemConfig::PAGE_SHIFT;
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

    constexpr static size_t kNumSizeClasses = SizeClass::Index(SizeConfig::MAX_TC_SIZE) + 1;
    std::array<SpanList, kNumSizeClasses> span_lists_{};
};


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
    AM_NODISCARD void* Allocate(size_t size) noexcept {
        if (size == 0) AM_UNLIKELY {
                return nullptr;
            }
        AM_DCHECK(size <= SizeConfig::MAX_TC_SIZE);

        size_t idx = SizeClass::Index(size);
        auto& list = free_lists_[idx];
        // 1. Fast Path: Pop from local free list (Lock-Free)
        if (!list.empty()) AM_LIKELY {
                return list.pop();
            }
        // 2. Slow Path: Fetch from CentralCache
        // Note: We must pass the aligned size to CentralCache/PageCache logic
        return FetchFromCentralCache(list, SizeClass::RoundUp(size));
    }

    /**
     * @brief Deallocate memory.
     * @param ptr Pointer to the memory.
     * @param size The size of the object (lookup via PageMap in global interface).
     */
    void Deallocate(void* ptr, size_t size) {
        AM_DCHECK(ptr != nullptr);
        AM_DCHECK(size <= SizeConfig::MAX_TC_SIZE);

        size_t idx = SizeClass::Index(size);
        auto& list = free_lists_[idx];
        // 1. Fast Path: Push to local free list (Lock-Free)
        list.push(ptr);

        // 2. Slow Path: Return memory if cache is too large (Scavenging)
        // If the list length exceeds the limit, return a batch to CentralCache.
        if (list.size() >= list.max_size()) {
            const auto limit = SizeClass::CalculateBatchSize(size);
            if (list.max_size() < limit) {
                list.set_max_size(list.max_size() + 1);
            } else {
                ReleaseTooLongList(list, size);
            }
        }
    }

    void ReleaseAll() {
        for (size_t i = 0; i < kNumSizeClasses; ++i) {
            auto& list = free_lists_[i];
            if (list.empty()) {
                continue;
            }

            const auto size = SizeClass::Size(i);
            void* start = nullptr;
            const auto cnt = list.size();
            for (size_t j = 0; j < cnt; ++j) {
                void* ptr = list.pop();
                static_cast<FreeBlock*>(ptr)->next = static_cast<FreeBlock*>(start);
                start = ptr;
            }
            CentralCache::GetInstance().ReleaseListToSpans(start, size);
        }
    }

private:
    // Size class configuration
    constexpr static size_t kNumSizeClasses = SizeClass::Index(SizeConfig::MAX_TC_SIZE) + 1;
    // Array of FreeLists. Access is lock-free as it's thread-local.
    std::array<FreeList, kNumSizeClasses> free_lists_{};

    /**
     * @brief Fetch objects from CentralCache when ThreadCache is empty.
     */
    static void* FetchFromCentralCache(FreeList& list, size_t size) {
        const auto limit = SizeClass::CalculateBatchSize(size);
        auto batch_num = list.max_size();
        if (batch_num > limit) {
            batch_num = limit;
        }

        // Fetch from CentralCache (This involves locking in CentralCache)
        // 'list' is modified in-place by FetchRange.
        auto actual_num = CentralCache::GetInstance().FetchRange(list, batch_num, size);
        if (actual_num == 0) {
            return nullptr;// Out of memory
        }

        // Dynamic Limit Strategy (Slow Start):
        if (list.max_size() < limit) {
            list.set_max_size(list.max_size() + 1);
        }
        return list.pop();
    }

    /**
     * @brief Return objects to CentralCache when ThreadCache is full.
     */
    static void ReleaseTooLongList(FreeList& list, size_t size) {
        // Strategy: When full, release 'batch_num' objects back to CentralCache.
        // This keeps 'batch_num' objects in ThreadCache (if limit is 2*batch),
        // or empties it if limit == batch.

        // Use the same batch calculation for releasing.
        // We pop 'batch_num' items from the list and link them together.
        auto batch_num = SizeClass::CalculateBatchSize(size);
        void* start = nullptr;
        // Construct a linked list of objects to return
        // We assume FreeList::pop() returns the raw pointer.
        // We use the object's memory to store the 'next' pointer (Embedded List).
        for (size_t i = 0; i < batch_num; ++i) {
            void* ptr = list.pop();
            // Link node: ptr->next = start; start = ptr;
            static_cast<FreeBlock*>(ptr)->next = static_cast<FreeBlock*>(start);
            start = ptr;
        }

        // Send the list to CentralCache
        CentralCache::GetInstance().ReleaseListToSpans(start, size);
    }
};
}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_MEMORY_POOL_H
