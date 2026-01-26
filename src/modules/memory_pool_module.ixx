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

struct FreeBlock {
    FreeBlock* next;
};

class FreeList {
public:
    FreeList() : head_(nullptr), size_(0) {}

    AM_NODISCARD bool empty() const {
        return head_ == nullptr;
    }

    AM_NODISCARD size_t size() const {
        return size_;
    }

    void clear() {
        head_ = nullptr;
        size_ = 0;
    }

    void push(void* ptr) {

    }

    void* pop() {
        if (empty()) {
            return nullptr;
        }

        auto* block = head_;
        head_ = head_->next;
        --size_;
        return block;
    }

private:
    FreeBlock* head_;
    size_t size_;
};

}// namespace aethermind