#include "aethermind/backend/backend_registry.h"
#include "aethermind/backend/backend.h"

namespace aethermind {

void BackendRegistry::SetFactory(DeviceType type, std::unique_ptr<BackendFactory> factory) {
    AM_CHECK(type != DeviceType::kUndefined, "Cannot register factory for kUndefined device type");
    AM_CHECK(factory != nullptr, "Backend factory cannot be null");
    factories_[type] = std::move(factory);

    std::erase_if(backends_, [type](const auto& entry) { return entry.first == type; });
}

StatusOr<Backend*> BackendRegistry::GetBackend(DeviceType type) noexcept {
    // Plan-build is currently single-threaded: factories are registered via
    // RuntimeBuilder before any plan build, and each device's backend is
    // created lazily at most once here. When multi-threaded plan-build lands,
    // guard both maps with a mutex (must be movable, since BackendRegistry is
    // moved into RuntimeContext).
    if (const auto it = backends_.find(type); it != backends_.end()) {
        return it->second.get();
    }

    const auto factory_it = factories_.find(type);
    if (factory_it == factories_.end()) {
        return Status::NotFound(
                std::format("No backend factory registered for device type: {}",
                            DeviceType2Str(type).c_str()));
    }

    auto backend_or = factory_it->second->Create();
    if (!backend_or.ok()) {
        return backend_or.status();
    }

    auto backend = std::move(backend_or).value();
    if (!backend) {
        return Status::Internal("Failed to create backend for device type: " +
                                std::string(DeviceType2Str(type).c_str()));
    }

    Backend* backend_ptr = backend.get();
    backends_[type] = std::move(backend);
    return backend_ptr;
}

} // namespace aethermind
