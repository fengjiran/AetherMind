//
// Created by 赵丹 on 2025/8/15.
//

#include "device.h"

namespace aethermind {

std::string DeviceType2Str(DeviceType device_type, bool lower_case) {
    switch (device_type) {
        case DeviceType::kCPU: {
            return lower_case ? "cpu" : "CPU";
        }

        case DeviceType::kCUDA: {
            return lower_case ? "cuda" : "CUDA";
        }

        case DeviceType::kCANN: {
            return lower_case ? "cann" : "CANN";
        }

        default: {
            AETHERMIND_THROW(runtime_error) << "Unsupported device type: " << static_cast<int>(device_type);
            AETHERMIND_UNREACHABLE();
        }
    }
}

bool isValidDeviceType(DeviceType device_type) {
    switch (device_type) {
        case kCPU:
        case kCUDA:
        case kCANN:
            return true;
        default:
            return false;
    }
}

std::ostream& operator<<(std::ostream& os, DeviceType device_type) {
    os << DeviceType2Str(device_type);
    return os;
}

}// namespace aethermind
