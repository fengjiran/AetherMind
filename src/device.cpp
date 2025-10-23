//
// Created by 赵丹 on 2025/8/15.
//

#include "device.h"
#include "container/string.h"
#include "error.h"

#include <glog/logging.h>

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
            AETHERMIND_THROW(runtime_error) << "Unsupported device type: " << static_cast<int>(device_type);
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

Device::Device(DeviceType type, int8_t index)
    : impl_(make_object<DeviceImpl>(type, index)) {}

Device::Device(ObjectPtr<DeviceImpl> impl)
    : impl_(std::move(impl)) {}


String Device::str() const {
    String s = DeviceType2Str(type(), true);
    if (has_index()) {
        s = s + ":" + std::to_string(index());
    }
    return s;
}

DeviceImpl* Device::get_impl_ptr_unsafe() const noexcept {
    return impl_.get();
}

DeviceImpl* Device::release_impl_unsafe() {
    return impl_.release();
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

// String Device::str() const {
//     String s = DeviceType2Str(type(), true);
//     if (has_index()) {
//         s = s + ":" + std::to_string(index());
//     }
//     return s;
// }
//
// void Device::validate() const {
//     CHECK(index() >= -1) << "Device index must be greater than or equal to -1, but got " << index();
//     CHECK(!is_cpu() || index() <= 0) << "CPU device index must be -1 or zero, but got " << index();
// }

std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.str();
    return os;
}

}// namespace aethermind
