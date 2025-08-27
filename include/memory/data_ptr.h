//
// Created by 赵丹 on 25-6-24.
//

#ifndef AETHERMIND_UNIQUE_VOID_PTR_H
#define AETHERMIND_UNIQUE_VOID_PTR_H

#include "macros.h"
#include "device.h"
#include <memory>

namespace aethermind {

using deleter_type = void (*)(void*);

inline void delete_nothing(void* ptr) {
    (void) ptr;
}

// This context is used to generate DataPtr which have arbitrary
// deleters associated with them. In some user facing functions,
// we give a (user-friendly) interface for constructing
// tensors from external data which take an arbitrary deleter.
// Grep for GeneralDataPtrContext to find these
// occurrences.
class DataPtrContext {
public:
    DataPtrContext(void* ptr, deleter_type deleter) : ptr_(ptr), deleter_(deleter) {}
    DataPtrContext(const DataPtrContext&) = delete;
    DataPtrContext& operator=(const DataPtrContext&) = delete;

    DataPtrContext(DataPtrContext&& other) noexcept
        : ptr_(other.ptr_), deleter_(other.deleter_) {
        other.ptr_ = nullptr;
        other.deleter_ = nullptr;
    }

    DataPtrContext& operator=(DataPtrContext&& other) noexcept {
        DataPtrContext(std::move(other)).swap(*this);
        return *this;
    }

    void swap(DataPtrContext& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(deleter_, other.deleter_);
    }

    NODISCARD deleter_type get_deleter() const {
        return deleter_;
    }

    void set_data_ptr(void* ptr) {
        ptr_ = ptr;
    }

    void set_deleter(deleter_type deleter) {
        deleter_ = deleter;
    }

    ~DataPtrContext() {
        if (deleter_ != nullptr) {
            deleter_(ptr_);
        }
    }

private:
    void* ptr_;
    deleter_type deleter_;
};

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
class DataPtr {
public:
    DataPtr(void* data, deleter_type deleter, Device device)
        : data_(data), ctx_(std::make_unique<DataPtrContext>(data, deleter)), device_(device) {}

    DataPtr() : DataPtr(nullptr, delete_nothing, Device(kUndefined)) {}

    NODISCARD Device device() const {
        return device_;
    }

    NODISCARD void* get() const {
        return data_;
    }

    NODISCARD deleter_type get_deleter() const {
        return ctx_->get_deleter();
    }

    NODISCARD DataPtrContext* get_context() const {
        return ctx_.get();
    }

    NODISCARD DataPtrContext* release_context() {
        return ctx_.release();
    }

    std::unique_ptr<DataPtrContext>&& move_context() {
        return std::move(ctx_);
    }

    operator bool() const {
        return data_ || ctx_;
    }

    bool compare_and_exchange_deleter(deleter_type expected_deleter, deleter_type new_deleter) const {
        if (get_deleter() != expected_deleter) {
            return false;
        }
        ctx_->set_deleter(new_deleter);
        return true;
    }

    void clear() {
        data_ = nullptr;
        ctx_.reset();
        device_ = Device(kUndefined);
    }

    void unsafe_set_device(Device device) {
        device_ = device;
    }

    NODISCARD bool unsafe_reset_data_and_ctx(void* new_data, deleter_type new_deleter) {
        if (get_deleter() != delete_nothing) {
            return false;
        }
        data_ = new_data;
        ctx_->set_data_ptr(new_data);
        ctx_->set_deleter(new_deleter);
        return true;
    }

private:
    void* data_;// The underlying data pointer.
    std::unique_ptr<DataPtrContext> ctx_;
    Device device_;
};

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
    return !dp;
}

inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
    return !dp;
}

inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
    return dp;
}

inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
    return dp;
}

}// namespace aethermind

#endif//AETHERMIND_UNIQUE_VOID_PTR_H
