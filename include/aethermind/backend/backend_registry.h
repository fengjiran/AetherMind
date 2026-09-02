#ifndef AETHERMIND_BACKEND_BACKEND_REGISTRY_H
#define AETHERMIND_BACKEND_BACKEND_REGISTRY_H

/// @file backend_registry.h
/// @brief Registry of backend factories and cached backend instances.
///
/// Owns per-`DeviceType` factories and lazily created backends. The registry
/// is the planning-time entry point for obtaining a `Backend` by device type.

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/base/device.h"
#include "aethermind/base/status.h"

#include <memory>
#include <unordered_map>

namespace aethermind {

/// @brief Registry of backend factories and cached backend instances.
///
/// Owns one factory per `DeviceType` and one lazily created backend per
/// device type. `GetBackend` creates the backend on first use via its
/// factory and caches it. `SetFactory` replaces the factory and evicts any
/// cached backend so the next lookup re-creates it. The registry is
/// move-only and not thread-safe; external synchronization is required for
/// concurrent access.
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
    /// the next `GetBackend` re-creates it from the new factory.
    ///
    /// @param type Device type to register. Must not be `kUndefined`.
    /// @param factory Factory to register. Must be non-null. Ownership is
    ///        transferred to the registry.
    void SetFactory(DeviceType type, std::unique_ptr<BackendFactory> factory);

    /// @brief Returns the backend for a device type, creating it if needed.
    ///
    /// Lazily creates the backend via its factory on first use and caches it.
    ///
    /// @param type Device type to lookup.
    /// @return Pointer to the cached backend on success, or an error when no
    ///         factory is registered or creation fails.
    StatusOr<Backend*> GetBackend(DeviceType type) noexcept;

private:
    /// Factories keyed by device type. Owned by the registry.
    std::unordered_map<DeviceType, std::unique_ptr<BackendFactory>> factories_;
    /// Cached backends keyed by device type. Created lazily via factories.
    std::unordered_map<DeviceType, std::unique_ptr<Backend>> backends_;
};

} // namespace aethermind
#endif
