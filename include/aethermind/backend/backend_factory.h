#ifndef AETHERMIND_BACKEND_BACKEND_FACTORY_H
#define AETHERMIND_BACKEND_BACKEND_FACTORY_H

#include "aethermind/backend/backend_fwd.h"
#include "device.h"
#include <memory>

namespace aethermind {

class BackendFactory {
public:
    virtual ~BackendFactory() = default;
    virtual DeviceType device_type() const noexcept = 0;
    virtual std::unique_ptr<Backend> Create() const = 0;
};

}// namespace aethermind
#endif
