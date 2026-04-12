//
// Created by 赵丹 on 25-1-23.
//

#ifndef AETHERMIND_MEMORY_ALLOCATOR_H
#define AETHERMIND_MEMORY_ALLOCATOR_H

#include "buffer.h"
#include "device.h"
#include "utils/thread_local_debug_info.h"

#include <memory>
#include <unordered_map>

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
    std::unordered_map<DeviceType, std::unique_ptr<AllocatorProvider>> providers_;
    std::unordered_map<Device, std::unique_ptr<Allocator>> instances_;
};

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
