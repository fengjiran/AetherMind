//
// Created by richard on 2/7/26.
//

#ifndef AETHERMIND_MALLOC_SPAN_H
#define AETHERMIND_MALLOC_SPAN_H

#include "ammalloc/common.h"
#include "ammalloc/config.h"
#include "utils/logging.h"

namespace aethermind {

/**
 * @brief Span represents a contiguous range of memory pages.
 */
struct Span {
    // --- Double linked list ---
    Span* next{nullptr};
    Span* prev{nullptr};

    // --- Page Cache Info ---
    size_t start_page_idx{0};// Global start page index
    size_t page_num{0};      // Number of contiguous pages

    // --- Central Cache Object Info ---
    size_t obj_size{0};// Size of objects allocated from this Span(if applicable)
    size_t use_count{0};
    size_t capacity{0};// Object capacity
    void* data_base_ptr{nullptr};

    // --- bitmap info ---
    uint64_t* bitmap{nullptr};
    size_t bitmap_num{0};
    size_t scan_cursor{0};

    // --- Status & Meta (Packed) ---
    bool is_used{false};// Is this span currently in CentralCache?

    uint64_t last_used_time_ms{0};// Last time this span was used in CentralCache (milliseconds since epoch)
    bool is_committed{true};      // Whether physical memory is committed (false means MADV_DONTNEED)

    Span() = default;
    Span(size_t start_page_idx_, size_t page_num_) noexcept
        : start_page_idx(start_page_idx_), page_num(page_num_) {}

    void Init(size_t object_size);

    // allocate an object
    void* AllocObject();
    /**
     * @brief Release an object back to this Span.
     * @note Must be called with the bucket lock held.
     */
    void FreeObject(void* ptr);

    AM_NODISCARD void* GetStartAddr() const noexcept {
        return details::PageIDToPtr(start_page_idx);
    }

    AM_NODISCARD void* GetEndAddr() const noexcept {
        const auto start = reinterpret_cast<uintptr_t>(GetStartAddr());
        constexpr size_t shift = std::countr_zero(SystemConfig::PAGE_SIZE);
        return reinterpret_cast<void*>(start + (page_num << shift));
    }
};

struct alignas(SystemConfig::CACHE_LINE_SIZE) SpanV2 {
    // --- 1. 链表指针 (16B) ---
    SpanV2* next{nullptr};
    SpanV2* prev{nullptr};

    // --- 2. 核心寻址与状态 (16B) ---
    uint64_t start_page_idx{0};// 8B: 保持全宽度，支持 sentinel (max)
    uint32_t page_num{0};      // 4B: 最大 40 亿页，足够了
    uint16_t flags{0};         // 2B: is_used, is_committed 等标志位
    uint16_t size_class_idx{0};// 2B: ThreadCache/CentralCache 的数组索引

    // --- 3. 对象分配元数据 (16B) ---
    uint32_t obj_size{0};   // 4B: 最大 4GB 对象
    uint32_t capacity{0};   // 4B: 最大 40 亿个对象
    uint32_t use_count{0};  // 4B: 当前使用量
    uint32_t scan_cursor{0};// 4B: 扫描游标

    // --- 4. 杂项与冷数据 (16B) ---
    uint32_t obj_offset{0};       // 4B: 替代 data_base_ptr，记录相对于 PageBase 的偏移
    uint32_t padding{0};          // 4B: 对齐填充
    uint64_t last_used_time_ms{0};// 8B: Scavenger 使用的时间戳

    enum FlagBit : uint16_t {
        kUsedMask = 1u << 0,
        kCommittedMask = 1u << 1
    };

    SpanV2() = default;
    SpanV2(uint64_t start_page_idx_, uint32_t page_num_) noexcept
        : start_page_idx(start_page_idx_), page_num(page_num_) {}

    AM_NODISCARD AM_ALWAYS_INLINE bool IsUsed() const noexcept {
        return flags & kUsedMask;
    }

    AM_NODISCARD AM_ALWAYS_INLINE bool IsCommitted() const noexcept {
        return flags & kCommittedMask;
    }

    AM_ALWAYS_INLINE void SetUsed(bool used) noexcept {
        flags = (flags & ~kUsedMask) | (used ? kUsedMask : 0);
    }

    AM_ALWAYS_INLINE void SetCommitted(bool committed) noexcept {
        flags = (flags & ~kCommittedMask) | (committed ? kCommittedMask : 0);
    }

    // ===================================================================
    // 内联计算方法
    // ===================================================================
    AM_NODISCARD AM_ALWAYS_INLINE void* GetPageBaseAddr() const noexcept {
        return reinterpret_cast<void*>(start_page_idx << SystemConfig::PAGE_SHIFT);
    }

    AM_NODISCARD AM_ALWAYS_INLINE uint64_t* GetBitmap() const noexcept {
        return static_cast<uint64_t*>(GetPageBaseAddr());
    }

    AM_NODISCARD AM_ALWAYS_INLINE size_t GetBitmapNum() const noexcept {
        return (capacity + 63) >> 6;
    }

    AM_NODISCARD AM_ALWAYS_INLINE void* GetDataBasePtr() const noexcept {
        return static_cast<char*>(GetPageBaseAddr()) + obj_offset;
    }

    void Init(size_t object_size);
    // allocate an object
    void* AllocObject();
    /**
     * @brief Release an object back to this Span.
     * @note Must be called with the bucket lock held.
     */
    void FreeObject(void* ptr);
};
static_assert(sizeof(SpanV2) == SystemConfig::CACHE_LINE_SIZE, "SpanV2 must be exactly 64 bytes");
static_assert(alignof(SpanV2) == SystemConfig::CACHE_LINE_SIZE, "SpanV2 alignment mismatch");

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
class alignas(SystemConfig::CACHE_LINE_SIZE) SpanList {
public:
    /// @brief Initializes an empty circular doubly linked list.
    /// The sentinel node's next and prev pointers point to itself.
    SpanList() noexcept {
        head_.next = &head_;
        head_.prev = &head_;
    }

    // Disable copy/move to prevent lock state corruption and pointer invalidation.
    SpanList(const SpanList&) = delete;
    SpanList& operator=(const SpanList&) = delete;

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

private:
    /**
     * @brief Sentinel node (Dummy Head).
     * Stored as a member object (not a pointer) to ensure it resides in the
     * same cache line as the SpanList object itself.
     */
    Span head_;
};


}// namespace aethermind

#endif// AETHERMIND_MALLOC_SPAN_H
