#ifndef AETHERMIND_BACKEND_BACKEND_REGISTRY_H
#define AETHERMIND_BACKEND_BACKEND_REGISTRY_H

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/base/device.h"
#include "aethermind/base/status.h"

#include <memory>
#include <unordered_map>

namespace aethermind {

class BackendRegistry {
public:
    BackendRegistry() = default;
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;
    BackendRegistry(BackendRegistry&&) noexcept = default;
    BackendRegistry& operator=(BackendRegistry&&) noexcept = default;
    ~BackendRegistry() = default;

    /// @brief Registers or replaces the factory for a device type.
    ///
    /// Calling again for the same type evicts any cached backend instance so
    /// the next GetBackend() re-creates it from the new factory.
    void SetFactory(DeviceType type, std::unique_ptr<BackendFactory> factory);
    StatusOr<Backend*> GetBackend(DeviceType type) noexcept;

private:
    std::unordered_map<DeviceType, std::unique_ptr<BackendFactory>> factories_;
    std::unordered_map<DeviceType, std::unique_ptr<Backend>> backends_;
};

} // namespace aethermind
#endif
