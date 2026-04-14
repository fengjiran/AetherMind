/// \file
/// Implementation of AllocatorRegistry and memory profiling utilities.
///
/// Provides:
/// - Provider registration and instance caching for AllocatorRegistry
/// - Thread-local memory profiling state check via memoryProfilingEnabled()

#include "aethermind/memory/allocator.h"
#include "utils/logging.h"

namespace aethermind {

void AllocatorRegistry::RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined, "Cannot register provider for kUndefined device type");
    AM_CHECK(!providers_.contains(type), "Allocator provider already registered for device type: {}", DeviceType2Str(type).c_str());
    providers_[type] = std::move(provider);
}

void AllocatorRegistry::SetProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined, "Cannot set provider for kUndefined device type");

    providers_[type] = std::move(provider);

    // Clear cached instances for the replaced provider's device type.
    // New instances will be created on next GetAllocator call.
    std::erase_if(instances_, [type](const auto& pair) { return pair.first.type() == type; });
}

Allocator& AllocatorRegistry::GetAllocator(Device device) {
    // TODO: Enable mutex for thread-safe access once multi-threaded runtime is supported.
    // std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(device);
    if (it != instances_.end()) {
        return *it->second;
    }

    auto type = device.type();
    AM_CHECK(providers_.contains(type), "No allocator provider registered for device type: {}",
             DeviceType2Str(type).c_str());

    auto allocator = providers_[type]->CreateAllocator(device);
    AM_CHECK(allocator != nullptr, "Allocator provider failed to create allocator for device: {}",
             device.ToString().c_str());
    AM_CHECK(allocator->device() == device, "Created allocator device mismatch: expected {}, got {}",
             device.ToString().c_str(), allocator->device().ToString().c_str());

    auto [new_it, inserted] = instances_.emplace(device, std::move(allocator));
    return *new_it->second;
}

bool memoryProfilingEnabled() {
    auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
            ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
    return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}


}// namespace aethermind
