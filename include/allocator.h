//
// Created by 赵丹 on 25-1-23.
//

#ifndef AETHERMIND_ALLOCATOR_H
#define AETHERMIND_ALLOCATOR_H

#include "env.h"
#include "tensor_utils.h"
#include "unique_void_ptr.h"

#include <glog/logging.h>
#include <unordered_map>

namespace aethermind {

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
//
class DataPtr {
public:
    DataPtr() : device_(DeviceType::kCPU) {}

    DataPtr(void* data, DeviceType device) : ptr_(data), device_(device) {}

    DataPtr(void* data, void* ctx, deleter_type deleter, DeviceType device)
        : ptr_(data, ctx, deleter), device_(device) {}

    NODISCARD DeviceType device() const {
        return device_;
    }

    void clear() {
        ptr_.clear();
    }

    NODISCARD void* get() const {
        return ptr_.get();
    }

    NODISCARD void* get_context() const {
        return ptr_.get_context();
    }

    NODISCARD deleter_type get_deleter() const {
        return ptr_.get_deleter();
    }

    void* release_context() {
        return ptr_.release_context();
    }

    std::unique_ptr<void, deleter_type>&& move_context() {
        return ptr_.move_context();
    }

    operator bool() const {
        return ptr_;
    }

    void* operator->() const {
        return ptr_.get();
    }

    bool compare_exchange_deleter(deleter_type expected_deleter, deleter_type new_deleter) {
        return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
    }

    template<typename T>
    T* cast_context(deleter_type expected_deleter) const {
        return ptr_.cast_context<T>(expected_deleter);
    }

    bool unsafe_reset_data_and_ctx(void* new_data_and_ctx) {
        return ptr_.unsafe_reset_data_and_ctx(new_data_and_ctx);
    }

    void unsafe_set_device(DeviceType device) {
        device_ = device;
    }

private:
    UniqueVoidPtr ptr_;
    DeviceType device_;
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

// template<typename Rollback>
// class ExceptionGuard {
// public:
//     ExceptionGuard(Rollback rollback) : rollback_(std::move(rollback)), complete_(false) {}
//
//     ExceptionGuard() = delete;
//     ExceptionGuard(const ExceptionGuard&) = delete;
//     ExceptionGuard(ExceptionGuard&&) noexcept = delete;
//     ExceptionGuard& operator=(const ExceptionGuard&) = delete;
//     ExceptionGuard& operator=(ExceptionGuard&&) noexcept = delete;
//
//     void complete() noexcept {
//         complete_ = true;
//     }
//
//     ~ExceptionGuard() {
//         if (!complete_) {
//             rollback_();
//         }
//     }
//
// private:
//     bool complete_;
//     Rollback rollback_;
// };


class Allocator {
public:
    Allocator() = default;
    virtual ~Allocator() = default;

    // NODISCARD virtual void* allocate(size_t nbytes) const = 0;

    NODISCARD virtual DataPtr allocate(size_t nbytes) const = 0;

    virtual void deallocate(void* p) const = 0;
};

class AllocatorTable {
public:
    static AllocatorTable& Global() {
        static AllocatorTable inst;
        return inst;
    }

    void set_allocator(DeviceType device, std::unique_ptr<Allocator> allocator) {
        table_[device] = std::move(allocator);
    }

    const std::unique_ptr<Allocator>& get_allocator(DeviceType device) {
        CHECK(table_.contains(device)) << "Allocator not found";
        return table_[device];
    }

private:
    AllocatorTable() = default;
    std::unordered_map<DeviceType, std::unique_ptr<Allocator>> table_;
};

#define REGISTER_ALLOCATOR(device, allocator)                                          \
    STR_CONCAT(REG_VAR_DEF, __COUNTER__) = [] {                                        \
        AllocatorTable::Global().set_allocator(device, std::make_unique<allocator>()); \
        return 0;                                                                      \
    }()


class CUDAAllocator : public Allocator {
public:
    CUDAAllocator() = default;
    // NODISCARD void* allocate(size_t n) const override {
    //     void* p = nullptr;
    //     // CHECK_CUDA(cudaMalloc(&p, n));
    //     return p;
    // }

    NODISCARD DataPtr allocate(size_t nbytes) const override {
        return {};
    }

    void deallocate(void* p) const override {
        // CHECK_CUDA(cudaFree(p));
    }
};

}// namespace aethermind

#endif//AETHERMIND_ALLOCATOR_H
