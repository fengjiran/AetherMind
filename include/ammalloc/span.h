//
// Created by richard on 2/7/26.
//

#ifndef AETHERMIND_MALLOC_SPAN_H
#define AETHERMIND_MALLOC_SPAN_H

#include "ammalloc/common.h"
#include "ammalloc/config.h"
#include "utils/logging.h"

#include <atomic>

namespace aethermind {

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
        void* start_ptr = details::PageIDToPtr(start_page_idx);
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
                details::CPUPause();
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
        return details::PageIDToPtr(start_page_idx);
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


}// namespace aethermind

#endif//AETHERMIND_MALLOC_SPAN_H
