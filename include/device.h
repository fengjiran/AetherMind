//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_DEVICE_H
#define AETHERMIND_DEVICE_H

#include "error.h"

#include <cstdint>
#include <string>

namespace aethermind {

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
    kCANN,
    kUndefined
};

constexpr DeviceType kCPU = DeviceType::kCPU;
constexpr DeviceType kCUDA = DeviceType::kCUDA;
constexpr DeviceType kCANN = DeviceType::kCANN;
constexpr DeviceType kUndefined = DeviceType::kUndefined;

std::string DeviceType2Str(DeviceType device_type, bool lower_case = false);

bool isValidDeviceType(DeviceType device_type);

std::ostream& operator<<(std::ostream& os, DeviceType device_type);

/// Represents a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
struct Device {
    Device() :Device(kCPU) {}

    explicit Device(DeviceType type, int8_t index = -1) : type_(type), index_(index) {
        validate();
    }

    NODISCARD DeviceType type() const noexcept {
        return type_;
    }

    NODISCARD int8_t index() const noexcept {
        return index_;
    }

    NODISCARD bool has_index() const noexcept {
        return index_ != -1;
    }

    NODISCARD bool is_cpu() const noexcept {
        return type_ == kCPU;
    }

    NODISCARD bool is_cuda() const noexcept {
        return type_ == kCUDA;
    }

    NODISCARD bool is_cann() const noexcept {
        return type_ == kCANN;
    }

    NODISCARD std::string str() const;

    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

private:
    DeviceType type_;
    int8_t index_ = -1;

    void validate() const;
};

std::ostream& operator<<(std::ostream& os, const Device& device);

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::DeviceType> {
    std::size_t operator()(aethermind::DeviceType k) const noexcept {
        return std::hash<int>()(static_cast<int>(k));
    }
};
}// namespace std

#endif//AETHERMIND_DEVICE_H
