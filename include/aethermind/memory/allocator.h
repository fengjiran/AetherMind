/// \file
/// Memory allocator abstraction and registry.
///
/// This file defines the core memory allocation interfaces for AetherMind:
/// - Allocator: Abstract interface for device-specific memory allocation
/// - AllocatorProvider: Factory interface for creating Allocator instances
/// - AllocatorRegistry: Global registry for allocator providers and instances
/// - MemoryReportingInfoBase: Interface for thread-local memory usage reporting
///
/// The allocation model uses Buffer as the primary memory container, which owns
/// the allocated memory through MemoryHandle. Allocators are registered per
/// DeviceType and instantiated per Device.

#ifndef AETHERMIND_MEMORY_ALLOCATOR_H
#define AETHERMIND_MEMORY_ALLOCATOR_H

#include "buffer.h"
#include "device.h"
#include "utils/thread_local_debug_info.h"

#include <memory>
#include <unordered_map>

namespace aethermind {

/// Abstract interface for device-specific memory allocation.
///
/// Allocators produce Buffer objects that own allocated memory via MemoryHandle.
/// Each Allocator is associated with a specific Device (e.g., CPU, CUDA).
///
/// Implementations must ensure:
/// - Allocate(0) returns a valid Buffer with nbytes() == 0
/// - The returned Buffer's MemoryHandle has a proper deleter that frees the memory
///
/// \note Allocator instances are owned by AllocatorRegistry. Do not create
///       Allocators directly; use AllocatorRegistry::GetAllocator().
class Allocator {
public:
    virtual ~Allocator() = default;

    /// Allocates a Buffer of the specified size.
    ///
    /// \param nbytes Number of bytes to allocate. May be zero.
    /// \return A Buffer owning the allocated memory.
    /// \throws RuntimeError if allocation fails.
    virtual Buffer Allocate(size_t nbytes) = 0;

    /// Returns the Device this allocator is associated with.
    /// \return The target Device for allocations from this allocator.
    AM_NODISCARD virtual Device device() const noexcept = 0;
};

/// Factory interface for creating Allocator instances.
///
/// AllocatorProviders are registered with AllocatorRegistry per DeviceType.
/// When GetAllocator is called for a Device, the corresponding Provider creates
/// an Allocator instance specific to that Device's index.
///
/// \note Implementations should cache device-specific state (e.g., CUDA streams)
///       per Device index if needed.
class AllocatorProvider {
public:
    virtual ~AllocatorProvider() = default;

    /// Creates an Allocator for the specified Device.
    ///
    /// \param device The Device to create an allocator for.
    /// \return A new Allocator instance for the device.
    /// \throws RuntimeError if the device type is unsupported or creation fails.
    virtual std::unique_ptr<Allocator> CreateAllocator(Device device) = 0;
};

/// Global registry for allocator providers and cached allocator instances.
///
/// AllocatorRegistry manages:
/// 1. Provider registration: One provider per DeviceType (CPU, CUDA, etc.)
/// 2. Instance caching: One Allocator per Device (type + index combination)
///
/// Call flow:
/// - RegisterProvider: Called during initialization to register providers
/// - GetAllocator: Called during runtime to obtain allocators for tensor allocation
///
/// \note This registry is not thread-safe. If concurrent access is needed,
///       external synchronization is required. The mutex_ member is commented
///       out pending multi-threaded runtime support.
class AllocatorRegistry {
public:
    /// Registers an allocator provider for a DeviceType.
    ///
    /// \param type The device type (kCPU, kCUDA, etc.). Must not be kUndefined.
    /// \param provider The provider to register. Must not be null.
    /// \throws RuntimeError if type is kUndefined or a provider is already registered.
    void RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);

    /// Sets or replaces an allocator provider for a DeviceType.
    ///
    /// Unlike RegisterProvider, this allows replacing an existing provider.
    /// Existing allocator instances for the DeviceType are cleared.
    ///
    /// \param type The device type. Must not be kUndefined.
    /// \param provider The provider to set. Must not be null.
    void SetProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);

    /// Returns the Allocator for a Device, creating it if necessary.
    ///
    /// If an allocator for the device already exists in the cache, returns it.
    /// Otherwise, creates a new allocator via the registered provider.
    ///
    /// \param device The Device to get an allocator for.
    /// \return Reference to the cached Allocator for this device.
    /// \throws RuntimeError if no provider is registered for the device type.
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
