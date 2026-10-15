/// \file
/// Validated Buffer allocation and device-specific allocator registry.

#include "aethermind/memory/allocator.h"
#include "aethermind/base/alignment.h"
#include "aethermind/base/error.h"
#include "utils/logging.h"

#include <cstdint>
#include <format>
#include <new>
#include <utility>

namespace aethermind {

StatusOr<Buffer> Allocator::TryAllocate(const AllocationRequest& request) noexcept try {
    if (request.alignment != 0 && !IsValidAlignment(request.alignment)) {
        return Status::InvalidArgument(
                "Allocation alignment must be zero or a power of two");
    }

    auto storage = AllocateStorage(request);
    if (!storage.ok()) {
        return storage.status();
    }

    MemoryHandle& owned = storage.value();
    if (!owned) {
        return Status::Internal(
                "Allocator storage hook returned an empty handle");
    }

    if (owned.deleter() == nullptr || owned.deleter() == &NoOpMemoryDeleter) {
        return Status::Internal(
                "Allocator storage hook returned borrowed memory");
    }

    const Device storage_device = owned.device();
    const DeviceType type = storage_device.type();
    if ((type != kCPU && type != kCUDA && type != kCANN) || storage_device != device()) {
        return Status::Internal(
                "Allocator storage hook returned an invalid or mismatched device");
    }

    const size_t alignment = owned.alignment();
    if (!IsValidAlignment(alignment) || alignment < request.alignment) {
        return Status::Internal(
                "Allocator storage hook returned an invalid or insufficient alignment");
    }

    if (type == kCPU && reinterpret_cast<uintptr_t>(owned.get()) % alignment != 0) {
        return Status::Internal(
                "Allocator storage pointer does not satisfy its declared alignment");
    }

    return Buffer{request.nbytes, std::move(owned)};
} catch (const std::bad_alloc&) {
    // Error diagnostics may allocate; reporting OOM must not allocate again.
    return Status::ResourceExhausted({});
}

Buffer Allocator::Allocate(size_t nbytes) {
    auto result = TryAllocate({.nbytes = nbytes});
    if (result.ok()) {
        return std::move(result).value();
    }

    const Status& status = result.status();
    if (status.code() == StatusCode::kResourceExhausted) {
        throw std::bad_alloc();
    }
    AM_THROW(RuntimeError) << "Allocation of " << nbytes << " bytes failed: "
                           << status.ToString();
    AM_UNREACHABLE();
}

void AllocatorRegistry::RegisterProvider(
        DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined,
             "Cannot register provider for kUndefined device type");
    AM_CHECK(!providers_.contains(type),
             "Allocator provider already registered for device type: {}",
             DeviceType2Str(type).c_str());
    providers_[type] = std::move(provider);
}

void AllocatorRegistry::SetProvider(
        DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined, "Cannot set provider for kUndefined device type");

    providers_[type] = std::move(provider);

    // Buffers retain their own storage lifetime when cached allocators go away.
    std::erase_if(instances_, [type](const auto& pair) {
        return pair.first.type() == type;
    });
}

Allocator& AllocatorRegistry::GetAllocator(Device device) {
    auto result = TryGetAllocator(device);
    AM_CHECK(result.ok(), "Failed to get allocator for device {}: {}",
             device.ToString().c_str(), result.status().ToString().c_str());
    return *result.value();
}

StatusOr<Allocator*> AllocatorRegistry::TryGetAllocator(Device device) {
    // TODO: Enable mutex for thread-safe access once multi-threaded runtime is supported.
    // std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(device);
    if (it != instances_.end()) {
        return it->second.get();
    }

    const auto type = device.type();
    if (!providers_.contains(type)) {
        return Status::FailedPrecondition(std::format(
                "No allocator provider registered for device type: {}",
                DeviceType2Str(type)));
    }

    auto allocator = providers_[type]->CreateAllocator(device);
    if (allocator == nullptr) {
        return Status::Internal(std::format(
                "Allocator provider failed to create allocator for device: {}",
                device.ToString()));
    }

    if (allocator->device() != device) {
        return Status::Internal(std::format(
                "Created allocator device mismatch: expected {}, got {}",
                device.ToString(), allocator->device().ToString()));
    }

    auto [new_it, inserted] = instances_.emplace(device, std::move(allocator));
    return new_it->second.get();
}

} // namespace aethermind
