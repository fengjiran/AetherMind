#include "aethermind/backend/backend_registry.h"
#include "aethermind/backend/backend.h"

#include <format>

namespace aethermind {

void BackendRegistry::RegisterFactory(DeviceType type, std::unique_ptr<BackendFactory> factory) {
    AM_CHECK(type != DeviceType::kUndefined, "Cannot register factory for kUndefined device type");
    AM_CHECK(factory != nullptr, "Backend factory cannot be null");
    AM_CHECK(!factories_.contains(type), "Backend factory already registered for device type: {}", DeviceType2Str(type).c_str());
    factories_[type] = std::move(factory);
}

void BackendRegistry::SetFactory(DeviceType type, std::unique_ptr<BackendFactory> factory) {
    AM_CHECK(type != DeviceType::kUndefined, "Cannot register factory for kUndefined device type");
    AM_CHECK(factory != nullptr, "Backend factory cannot be null");
    factories_[type] = std::move(factory);

    for (auto it = backends_.begin(); it != backends_.end();) {
        if (it->first == type) {
            it = backends_.erase(it);
            continue;
        }
        ++it;
    }
}

StatusOr<Backend*> BackendRegistry::GetBackend(DeviceType type) {
    // TODO: Enable mutex for thread-safe access once multi-threaded runtime is supported.
    // std::lock_guard<std::mutex> lock(mutex_);
    if (const auto it = backends_.find(type); it != backends_.end()) {
        return it->second.get();
    }

    const auto factory_it = factories_.find(type);
    if (factory_it == factories_.end()) {
        return Status::NotFound(
                std::format("No backend factory registered for device type: {}",
                            DeviceType2Str(type).c_str()));
    }

    auto backend = factory_it->second->Create();
    if (!backend) {
        return Status::Internal(
                std::format("Failed to create backend for device type: {}",
                            DeviceType2Str(type).c_str()));
    }

    Backend* backend_ptr = backend.get();
    backends_[type] = std::move(backend);
    return backend_ptr;
}

}// namespace aethermind
