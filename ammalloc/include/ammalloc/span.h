#ifndef AETHERMIND_MALLOC_SPAN_H
#define AETHERMIND_MALLOC_SPAN_H

#include "ammalloc/common.h"
#include "ammalloc/config.h"
#include "utils/logging.h"

namespace aethermind {

/// Continuous page range metadata for ammalloc.
///
/// A Span represents a contiguous range of system pages allocated from the OS.
/// It is used by PageCache for coarse-grained memory management and by
/// CentralCache for fine-grained object allocation.
///
/// Memory layout is cache-line aligned (64B) to prevent false sharing
/// between threads accessing different fields.
///
/// Thread-safety: Individual fields are not thread-safe. Concurrent access
/// must be protected by the appropriate cache lock (PageCache::mutex_ or
/// CentralCache bucket locks).
struct alignas(SystemConfig::CACHE_LINE_SIZE) Span {
    // Intrusive linked list pointers for PageCache/SpanList.
    Span* next{nullptr};
    Span* prev{nullptr};

    // Page-level addressing info.
    uint64_t start_page_idx{0};// Supports sentinel values (e.g., max)
    uint32_t page_num{0};

    // Packed status flags. Use IsUsed/SetUsed/IsCommitted/SetCommitted.
    uint16_t flags{0};
    uint16_t size_class_idx{0};// Index into CentralCache bucket array

    // Object allocation metadata (valid when used by CentralCache).
    uint32_t obj_size{0};
    uint32_t capacity{0};   // Maximum objects storable in this span
    uint32_t use_count{0};  // Currently allocated objects
    uint32_t scan_cursor{0};// Bitmap search optimization

    // Calculated data offset (avoids storing full pointer).
    uint32_t obj_offset{0};// Offset from page base to first object
    uint32_t padding{0};   // Cache line alignment

    // Cold data: used by background scavenger thread.
    uint64_t last_used_time_ms{0};

    enum FlagBit : uint16_t {
        kUsedMask = 1u << 0,
        kCommittedMask = 1u << 1
    };

    Span() = default;
    Span(uint64_t start_page_idx_, uint32_t page_num_) noexcept
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

    // Bitmap and data address are calculated from page base to save space.
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

    /// Initialize span for object allocation with the given size.
    void Init(size_t object_size);

    /// Allocate an object from this span.
    /// @return nullptr if span is full.
    void* AllocObject();

    /// Release an object back to this span.
    /// @pre Must be called with the CentralCache bucket lock held.
    void FreeObject(void* ptr);
};
static_assert(sizeof(Span) == SystemConfig::CACHE_LINE_SIZE, "Span must be exactly 64 bytes");
static_assert(alignof(Span) == SystemConfig::CACHE_LINE_SIZE, "Span alignment mismatch");

/// Intrusive doubly linked list of Spans using a circular sentinel.
///
/// Thread-safety: Not thread-safe. External synchronization required.
/// CentralCache uses SpanList with its own bucket lock.
///
/// Design notes:
/// - Circular sentinel eliminates null checks in insert/erase
/// - Cache-line aligned to prevent false sharing with adjacent data
/// - Lifetime managed by PageCache (erase does not delete)
class alignas(SystemConfig::CACHE_LINE_SIZE) SpanList {
public:
    /// Creates empty list with circular sentinel.
    SpanList() noexcept {
        head_.next = &head_;
        head_.prev = &head_;
    }

    SpanList(const SpanList&) = delete;
    SpanList& operator=(const SpanList&) = delete;

    AM_NODISCARD Span* begin() noexcept { return head_.next; }
    AM_NODISCARD Span* end() noexcept { return &head_; }
    AM_NODISCARD bool empty() const noexcept { return head_.next == &head_; }

    /// Inserts span before pos.
    /// @pre Caller must hold the bucket lock.
    /// @pre pos != nullptr, span != nullptr
    static void insert(Span* pos, Span* span) noexcept {
        AM_DCHECK(pos != nullptr && span != nullptr);
        span->next = pos;
        span->prev = pos->prev;
        span->prev->next = span;
        pos->prev = span;
    }

    /// LIFO insert at front (improves cache locality).
    void push_front(Span* span) noexcept { insert(begin(), span); }

    /// LIFO insert at back.
    void push_back(Span* span) noexcept { insert(end(), span); }

    /// Removes span from list. Does NOT delete the Span.
    /// @pre Caller must hold the bucket lock.
    /// @return Next node in list.
    Span* erase(Span* span) noexcept {
        AM_DCHECK(span != nullptr && span != &head_);
        auto* prev = span->prev;
        auto* next = span->next;
        prev->next = next;
        next->prev = prev;
        span->prev = nullptr;
        span->next = nullptr;
        return next;
    }

    /// Removes first span. Returns nullptr if empty.
    Span* pop_front() noexcept {
        if (empty()) AM_UNLIKELY {
                return nullptr;
            }
        auto* span = head_.next;
        erase(span);
        return span;
    }

private:
    // Sentinel node stored inline to ensure cache line locality.
    Span head_;
};


}// namespace aethermind

#endif// AETHERMIND_MALLOC_SPAN_H
