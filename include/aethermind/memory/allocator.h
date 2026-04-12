//
// Created by 赵丹 on 25-1-23.
//

#ifndef AETHERMIND_MEMORY_ALLOCATOR_H
#define AETHERMIND_MEMORY_ALLOCATOR_H

#include "buffer.h"
#include "data_ptr.h"
#include "device.h"
#include "utils/thread_local_debug_info.h"

#include <memory>
#include <unordered_map>
// #include <mutex>

namespace aethermind {

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual Buffer Allocate(size_t nbytes) = 0;
    AM_NODISCARD virtual Device device() const noexcept = 0;
};

class AllocatorProvider {
public:
    virtual ~AllocatorProvider() = default;
    virtual std::unique_ptr<Allocator> CreateAllocator(Device device) = 0;
};

class AllocatorRegistry {
public:
    void RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);
    void SetProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);
    Allocator& GetAllocator(Device device);

private:
    // std::mutex mutex_;
    std::unordered_map<DeviceType, std::unique_ptr<AllocatorProvider>> providers_;
    std::unordered_map<Device, std::unique_ptr<Allocator>> instances_;
};


// =============================================================================
// Legacy Allocator API (Deprecated - Migration Only)
// =============================================================================
// The following classes are DEPRECATED and will be removed in a future release.
// Use Allocator/AllocatorRegistry/AllocatorProvider instead.
//
// These are maintained only for backward compatibility during the Storage-Tensor
// to Buffer-Tensor migration. New code MUST NOT use these APIs.

class [[deprecated("Use Allocator/AllocatorRegistry instead")]] AllocatorBK {
public:
    AllocatorBK() = default;
    virtual ~AllocatorBK() = default;

    AM_NODISCARD virtual DataPtr allocate(size_t nbytes) const = 0;

    virtual void deallocate(void* p) const = 0;
};

class [[deprecated("Use AllocatorRegistry instead")]] AllocatorTable {
public:
    [[deprecated("Use AllocatorRegistry instead")]]
    static AllocatorTable& Global() {
        alignas(alignof(AllocatorTable)) static char storage[sizeof(AllocatorTable)];
        static auto* inst = new (storage) AllocatorTable();
        return *inst;
    }

    [[deprecated("Use AllocatorRegistry::RegisterProvider instead")]]
    void set_allocator(DeviceType device, std::unique_ptr<AllocatorBK> allocator) {
        table_[device] = std::move(allocator);
    }

    [[deprecated("Use AllocatorRegistry::GetAllocator instead")]]
    const std::unique_ptr<AllocatorBK>& get_allocator(DeviceType device) {
        AM_CHECK(table_.contains(device), "Allocator not found");
        return table_[device];
    }

private:
    AllocatorTable() = default;
    std::unordered_map<DeviceType, std::unique_ptr<AllocatorBK>> table_;
};

class UndefinedAllocator final : public AllocatorBK {
public:
    UndefinedAllocator() = default;

    AM_NODISCARD DataPtr allocate(size_t nbytes) const override {
        return {};
    }

    void deallocate(void* p) const override {}
};

// DEPRECATED: Use AllocatorRegistry::RegisterProvider instead.
// This macro is retained only for legacy code during migration.
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

#endif// AETHERMIND_MEMORY_ALLOCATOR_H
