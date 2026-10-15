/// \file
/// Memory allocation interfaces and allocator registry.
///
/// Allocators validate allocation requests, obtain device-specific storage,
/// and transfer its MemoryHandle directly into a shared Buffer.

#ifndef AETHERMIND_MEMORY_ALLOCATOR_H
#define AETHERMIND_MEMORY_ALLOCATOR_H

#include "aethermind/base/device.h"
#include "aethermind/base/status.h"
#include "buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace aethermind {

/// @brief Visible storage size and minimum alignment required by a caller.
///
/// Layout padding must already be included in nbytes. The allocator decides
/// how to obtain storage and may provide stronger alignment than requested.
struct AllocationRequest {
    size_t nbytes = 0;

    /// Zero selects the allocator's default policy; a non-zero minimum must
    /// be a power of two. Subviews must account for their own byte offsets.
    size_t alignment = 0;
};

/// @brief Device-specific allocation through a common validated entry point.
///
/// TryAllocate validates both the request and the storage hook's result, then
/// transfers the original MemoryHandle into Buffer without wrapping its deleter.
/// The handle must own its storage independently of the allocator's lifetime.
class Allocator {
public:
    virtual ~Allocator() = default;

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    /// @brief Allocates an owning Buffer satisfying the requested size/alignment.
    /// @return Buffer on success; InvalidArgument for malformed alignment;
    ///         Internal for invalid hook ownership, device, or alignment;
    ///         the hook's failure otherwise. Storage or Buffer metadata OOM
    ///         returns ResourceExhausted with an empty message. Failures release
    ///         any owning storage already obtained.
    StatusOr<Buffer> TryAllocate(const AllocationRequest& request) noexcept;

    /// @brief Allocates using the default policy through TryAllocate.
    /// @throws std::bad_alloc for ResourceExhausted, Error for other failures.
    Buffer Allocate(size_t nbytes);

    AM_NODISCARD virtual Device device() const noexcept = 0;

protected:
    Allocator() noexcept = default;

    /// @brief Obtains storage for a validated request, including zero-byte ones.
    ///
    /// The handle must be non-null, owning, and declare this allocator's device
    /// and a known alignment satisfying the minimum. Its deleter must not borrow
    /// an allocator or other resource that can expire before the Buffer.
    /// Report failures through Status; std::bad_alloc is converted by TryAllocate.
    virtual StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) = 0;
};

/// @brief Factory for allocator instances associated with a Device.
class AllocatorProvider {
public:
    virtual ~AllocatorProvider() = default;
    virtual std::unique_ptr<Allocator> CreateAllocator(Device device) = 0;
};

/// @brief Owns providers by DeviceType and caches allocator instances by Device.
///
/// This registry is not thread-safe; callers must serialize concurrent access.
/// Buffers keep their own storage ownership and may outlive cached allocators.
class AllocatorRegistry {
public:
    AllocatorRegistry() = default;

    /// @brief Registers a new provider. Invalid/null/duplicate registration aborts.
    void RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);

    /// @brief Replaces a provider and discards its cached allocator instances.
    /// Existing Buffers remain owned by their original MemoryHandles.
    void SetProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);

    /// @brief Returns the cached allocator, creating it on first use.
    /// @note Aborts if lookup or provider validation returns a failure status.
    Allocator& GetAllocator(Device device);

    /// @brief Returns the cached allocator or reports an unavailable provider.
    /// @return Borrowed pointer, FailedPrecondition for a missing provider,
    ///         or Internal for a null/device-mismatched provider result.
    /// @note Provider exceptions and cache allocation failures propagate.
    StatusOr<Allocator*> TryGetAllocator(Device device);

    AM_NODISCARD bool HasProvider(DeviceType type) const noexcept {
        return providers_.contains(type);
    }

private:
    std::unordered_map<DeviceType, std::unique_ptr<AllocatorProvider>> providers_;
    std::unordered_map<Device, std::unique_ptr<Allocator>> instances_;
};

} // namespace aethermind

#endif // AETHERMIND_MEMORY_ALLOCATOR_H
