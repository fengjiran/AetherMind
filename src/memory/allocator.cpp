//
// Created by 赵丹 on 25-1-23.
//

#include "aethermind/memory/allocator.h"
#include "utils/logging.h"

namespace aethermind {

void AllocatorRegistry::RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined, "Cannot register provider for kUndefined device type");
    AM_CHECK(!providers_.contains(type), "Allocator provider already registered for device type: {}", DeviceType2Str(type).c_str());
    providers_[type] = std::move(provider);
}

Allocator& AllocatorRegistry::GetAllocator(Device device) {
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

REGISTER_ALLOCATOR(DeviceType::kUndefined, UndefinedAllocator);

bool memoryProfilingEnabled() {
    auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
            ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
    return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}


}// namespace aethermind
