//
// Created by 赵丹 on 2025/8/15.
//

#include "device.h"
#include "error.h"

#include <charconv>
#include <limits>

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
            AM_UNREACHABLE();
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

bool IsDeviceKindSupported(DeviceType device_type) {
    return device_type == DeviceType::kCPU;
}

StatusOr<Device> Device::Make(DeviceType type, int8_t index) {
    if (!IsValidDeviceType(type) || type == DeviceType::kUndefined) {
        return Status::InvalidArgument("Unsupported device type");
    }
    if (index < -1) {
        return Status::InvalidArgument("Device index must be greater than or equal to -1");
    }
    if (type == DeviceType::kCPU && index > 0) {
        return Status::InvalidArgument("CPU device index must be -1 or 0");
    }
    return Device(type, index);
}

StatusOr<Device> Device::FromString(std::string_view text) {
    if (text.empty()) {
        return Status::InvalidArgument("Device string must not be empty");
    }

    if (text == std::string_view("cpu")) {
        return Device::Make(DeviceType::kCPU, -1);
    }
    if (text == std::string_view("cuda")) {
        return Device::Make(DeviceType::kCUDA, -1);
    }
    if (text == std::string_view("cann")) {
        return Device::Make(DeviceType::kCANN, -1);
    }

    size_t colon = text.find(':');
    if (colon == std::string_view::npos) {
        return Status::InvalidArgument("Unsupported device string format");
    }

    std::string_view kind_part = text.substr(0, colon);
    std::string_view index_part = text.substr(colon + 1);

    DeviceType type = DeviceType::kUndefined;
    if (kind_part == std::string_view("cpu")) {
        type = DeviceType::kCPU;
    } else if (kind_part == std::string_view("cuda")) {
        type = DeviceType::kCUDA;
    } else if (kind_part == std::string_view("cann")) {
        type = DeviceType::kCANN;
    } else {
        return Status::InvalidArgument("Unsupported device kind");
    }

    if (index_part.empty()) {
        return Status::InvalidArgument("Device index is missing");
    }

    int parsed = 0;
    const char* begin = index_part.data();
    const char* end = index_part.data() + index_part.size();
    std::from_chars_result result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc() || result.ptr != end) {
        return Status::InvalidArgument("Device index must be an integer");
    }
    if (parsed < -1 || parsed > std::numeric_limits<int8_t>::max()) {
        return Status::InvalidArgument("Device index is out of range");
    }

    return Device::Make(type, static_cast<int8_t>(parsed));
}

String Device::ToString() const {
    String s = DeviceType2Str(type(), true);
    if (has_index()) {
        s += ":";
        s += std::to_string(index());
    }
    return s;
}


String Device::str() const {
    return ToString();
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
