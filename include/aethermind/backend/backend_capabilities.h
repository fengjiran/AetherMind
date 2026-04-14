#ifndef AETHERMIND_BACKEND_BACKEND_CAPABILITIES_H
#define AETHERMIND_BACKEND_BACKEND_CAPABILITIES_H

#include "device.h"

namespace aethermind {

struct BackendCapabilities {
    DeviceType device_type = DeviceType::kUndefined;
};

}// namespace aethermind
#endif
