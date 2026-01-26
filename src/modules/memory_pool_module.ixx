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
    // Use dynamic Batch logic: take more for small chunks, take fewer for large chunks
    static constexpr size_t CalculateBatchSize(size_t size) {
        if (size == 0) return 0;
        size_t batch = MAX_TC_SIZE / size;
        if (batch < 2) batch = 2;    // at least 2
        if (batch > 128) batch = 128;// At most 128, to prevent the central pool from being drained instantly
        return batch;
    }

    constexpr static size_t BATCH_SIZE = 20;
    // Maximum number of consecutive pages managed by Page Cache
    // (to avoid excessively large Spans)
    constexpr static size_t MAX_PAGE_NUM = 1024;
};

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

}// namespace aethermind