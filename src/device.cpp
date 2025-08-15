//
// Created by 赵丹 on 2025/8/15.
//

#include "device.h"

#include <glog/logging.h>

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
    os << DeviceType2Str(device_type, true);
    return os;
}

std::string Device::str() const {
    std::string s = DeviceType2Str(type(), true);
    if (has_index()) {
        s += ":" + std::to_string(index());
    }
    return s;
}

void Device::validate() const {
    CHECK(index() >= -1) << "Device index must be greater than or equal to -1, but got " << index();
    CHECK(!is_cpu() || index() <= 0) << "CPU device index must be -1 or zero, but got " << index();
}

std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.str();
    return os;
}

}// namespace aethermind
