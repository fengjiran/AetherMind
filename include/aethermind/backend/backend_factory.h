#ifndef AETHERMIND_BACKEND_BACKEND_FACTORY_H
#define AETHERMIND_BACKEND_BACKEND_FACTORY_H

/// @file backend_factory.h
/// @brief Factory for creating device-specific backends.
///
/// Defines the `BackendFactory` interface that creates a `Backend` for a
/// given `DeviceType`. Capability detection or device initialization failures
/// are reported via `StatusOr` so `BackendRegistry` can propagate them.

#include "aethermind/backend/backend_fwd.h"
#include "aethermind/base/device.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"

#include <memory>

namespace aethermind {

/// @brief Factory that creates a `Backend` for a specific device type.
///
/// Each factory handles one `DeviceType`. Creation may perform capability
/// detection or device initialization; failures are returned as `Status`
/// rather than aborting, allowing the registry to propagate them.
class BackendFactory {
public:
    virtual ~BackendFactory() = default;

    /// @brief Returns the device type handled by this factory.
    ///
    /// @return Device type (e.g. `kCPU`).
    AM_NODISCARD virtual DeviceType device_type() const noexcept = 0;

    /// @brief Creates a backend instance for the factory device type.
    ///
    /// Errors (e.g. capability detection or device initialization failures)
    /// are reported through the returned status instead of aborting, letting
    /// `BackendRegistry` propagate them to callers.
    ///
    /// @return Backend instance on success, or an error status on failure.
    AM_NODISCARD virtual StatusOr<std::unique_ptr<Backend>> Create() const = 0;
};

} // namespace aethermind
#endif
