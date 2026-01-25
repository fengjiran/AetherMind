//
// Created by 赵丹 on 2025/8/15.
//

#include "device.h"
#include "error.h"

namespace aethermind {

String DeviceType2Str(DeviceType device_type, bool lower_case) {
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
            AM_THROW(runtime_error) << "Unsupported device type: " << static_cast<int>(device_type);
            AETHERMIND_UNREACHABLE();
        }
    }
}

bool IsValidDeviceType(DeviceType device_type) {
    switch (device_type) {
        case kCPU:
        case kCUDA:
        case kCANN:
            return true;
        default:
            return false;
    }
}


String Device::str() const {
    String s = DeviceType2Str(type(), true);
    if (has_index()) {
        // s = s + ":" + std::to_string(index());
        s += ":";
        s += std::to_string(index());
    }
    return s;
}

Device Device::CPU() {
    return Device(kCPU);
}

Device Device::CUDA() {
    return Device(kCUDA);
}

Device Device::CANN() {
    return Device(kCANN);
}

std::ostream& operator<<(std::ostream& os, DeviceType device_type) {
    os << DeviceType2Str(device_type, true);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.str();
    return os;
}

}// namespace aethermind
