//
// Created by 赵丹 on 25-1-23.
//

#ifndef AETHERMIND_ALLOCATOR_H
#define AETHERMIND_ALLOCATOR_H

#include "buffer.h"
#include "container/map.h"
#include "data_ptr.h"
#include "device.h"
#include "utils/thread_local_debug_info.h"

#include <unordered_map>

namespace aethermind {

class IAllocator {
public:
    IAllocator() = default;
    IAllocator(const IAllocator&) = default;
    IAllocator(IAllocator&&) noexcept = default;

    IAllocator& operator=(const IAllocator&) = default;
    IAllocator& operator=(IAllocator&&) noexcept = default;

    virtual ~IAllocator() = default;
    virtual Buffer Allocate(size_t nbytes, size_t alignment) = 0;
    AM_NODISCARD virtual Device device() const noexcept = 0;
};

class IAllocatorProvider {
public:
    virtual ~IAllocatorProvider() = default;
    virtual std::shared_ptr<IAllocator> Get(Device device) = 0;
};

class AllocatorRegistry {
public:
    void Register(DeviceType type, std::unique_ptr<IAllocatorProvider> provider);
    IAllocatorProvider& GetProvider(DeviceType type) const;
private:
    std::unordered_map<DeviceType, std::unique_ptr<IAllocatorProvider>> providers_;
};

class RuntimeContext {
public:
    IAllocator& GetAllocator(Device device);
private:
    AllocatorRegistry registry_;
};

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

    AM_NODISCARD virtual DataPtr allocate(size_t nbytes) const = 0;

    virtual void deallocate(void* p) const = 0;
};

class AllocatorTable {
public:
    static AllocatorTable& Global() {
        alignas(alignof(AllocatorTable)) static char storage[sizeof(AllocatorTable)];
        static auto* inst = new (storage) AllocatorTable();
        return *inst;
    }

    void set_allocator(DeviceType device, std::unique_ptr<Allocator> allocator) {
        table_[device] = std::move(allocator);
    }

    const std::unique_ptr<Allocator>& get_allocator(DeviceType device) {
        AM_CHECK(table_.contains(device), "Allocator not found");
        return table_[device];
    }

private:
    AllocatorTable() = default;
    std::unordered_map<DeviceType, std::unique_ptr<Allocator>> table_;
    // Map<DeviceType, std::unique_ptr<Allocator>> table_;
};

class UndefinedAllocator final : public Allocator {
public:
    UndefinedAllocator() = default;

    AM_NODISCARD DataPtr allocate(size_t nbytes) const override {
        return {};
    }

    void deallocate(void* p) const override {}
};

class CUDAAllocator final : public Allocator {
public:
    CUDAAllocator() = default;
    // NODISCARD void* allocate(size_t n) const override {
    //     void* p = nullptr;
    //     // CHECK_CUDA(cudaMalloc(&p, n));
    //     return p;
    // }

    AM_NODISCARD DataPtr allocate(size_t nbytes) const override {
        return {};
    }

    void deallocate(void* p) const override {
        // CHECK_CUDA(cudaFree(p));
    }
};

#define REGISTER_ALLOCATOR(device, allocator)                                          \
    STR_CONCAT(REG_VAR_DEF, __COUNTER__) = [] {                                        \
        AllocatorTable::Global().set_allocator(device, std::make_unique<allocator>()); \
        return 0;                                                                      \
    }()

// An interface for reporting thread local memory usage per device
class MemoryReportingInfoBase : public DebugInfoBase {
public:
    /**
     * \brief Reports memory usage.
     * \param ptr The pointer to the memory block.
     *
     * \param alloc_size The size of the memory block.
     *
     * \param total_allocated The total allocated memory.
     *
     * \param total_reserved The total reserved memory.
     *
     * \param device The device type.
     */
    virtual void reportMemoryUsage(void* ptr,
                                   int64_t alloc_size,
                                   size_t total_allocated,
                                   size_t total_reserved,
                                   Device device) = 0;

    virtual void reportOutOfMemory(int64_t alloc_size,
                                   size_t total_allocated,
                                   size_t total_reserved,
                                   Device device);

    virtual bool memoryProfilingEnabled() const = 0;
};

bool memoryProfilingEnabled();

}// namespace aethermind

#endif// AETHERMIND_ALLOCATOR_H
