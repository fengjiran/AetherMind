#ifndef AETHERMIND_BACKEND_CPU_CPU_CAPABILITIES_H
#define AETHERMIND_BACKEND_CPU_CPU_CAPABILITIES_H

#include "aethermind/backend/backend_capabilities.h"

namespace aethermind {

struct CpuCapabilities {
    BackendCapabilities base{
            .device_type = DeviceType::kCPU};
    bool supports_inline_execution = true;
};

}// namespace aethermind
#endif
