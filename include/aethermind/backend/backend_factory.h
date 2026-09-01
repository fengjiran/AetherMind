#ifndef AETHERMIND_BACKEND_BACKEND_FACTORY_H
#define AETHERMIND_BACKEND_BACKEND_FACTORY_H

#include "aethermind/backend/backend_fwd.h"
#include "aethermind/base/device.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"

#include <memory>

namespace aethermind {

class BackendFactory {
public:
    virtual ~BackendFactory() = default;
    AM_NODISCARD virtual DeviceType device_type() const noexcept = 0;
    /// @brief Creates a backend instance for the factory device type.
    ///
    /// Errors (e.g. capability detection or device initialization failures)
    /// are reported through the returned Status instead of aborting, letting
    /// BackendRegistry propagate them to callers.
    AM_NODISCARD virtual StatusOr<std::unique_ptr<Backend>> Create() const = 0;
};

} // namespace aethermind
#endif
