#ifndef AETHERMIND_BASE_DEVICE_H
#define AETHERMIND_BASE_DEVICE_H

/// @file device.h
/// @brief Compute device abstraction for tensor placement and kernel dispatch.
///
/// A Device uniquely identifies a compute resource by type (CPU/CUDA/CANN) and
/// an optional index. This module provides:
/// - DeviceType enum and Device class for device identification
/// - Factory methods with validation (Make, FromString)
/// - Query helpers (IsDeviceKindSupported, DeviceType2Str)
///
/// Thread-safety: Device is immutable and thread-safe after construction.

#include "aethermind/base/status.h"
#include "utils/logging.h"

#include <string>

namespace aethermind {

/// @brief Compute device types supported by the runtime.
enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA,
    kCANN,
    kUndefined
};

/// @brief Convenience aliases for DeviceType enum values.
constexpr auto kCPU = DeviceType::kCPU;
constexpr auto kCUDA = DeviceType::kCUDA;
constexpr auto kCANN = DeviceType::kCANN;
constexpr auto kUndefined = DeviceType::kUndefined;

/// @brief Represents a compute device for tensor placement and kernel dispatch.
///
/// A device is identified by:
/// - type: The device kind (CPU, CUDA, CANN)
/// - index: Optional device ordinal (-1 means "current device")
///
/// Invariants:
/// - index must be >= -1
/// - CPU devices must have index -1 or 0 (no multi-CPU support)
///
/// Thread-safety: Immutable after construction; safe for concurrent read.
///
/// Example:
///   Device d = Device::CPU();           // Default CPU device
///   auto d2 = Device::Make(kCUDA, 0);   // Returns StatusOr, validates
///   auto d3 = Device::FromString("cuda:1");  // Parse from string
class Device {
public:
    /// @brief Constructs a default CPU device with index -1.
    Device() : Device(DeviceType::kCPU) {}

    /// @brief Constructs a device with explicit type and optional index.
    ///
    /// @param type  Device type (CPU, CUDA, CANN).
    /// @param index Device ordinal; -1 means "current device".
    /// @throws std::runtime_error if validation fails (via AM_CHECK).
    explicit Device(DeviceType type, int8_t index = -1) : type_(type), index_(index) {
        validate();
    }

    AM_NODISCARD DeviceType type() const noexcept {
        return type_;
    }

    AM_NODISCARD int8_t index() const noexcept {
        return index_;
    }

    /// @brief Returns true if an explicit index was provided (index >= 0).
    AM_NODISCARD bool has_index() const noexcept {
        return index_ != -1;
    }

    AM_NODISCARD bool is_cpu() const noexcept {
        return type_ == kCPU;
    }

    AM_NODISCARD bool is_cuda() const noexcept {
        return type_ == kCUDA;
    }

    AM_NODISCARD bool is_cann() const noexcept {
        return type_ == kCANN;
    }

    /// @brief Returns the string representation, e.g., "cpu" or "cuda:0".
    ///
    /// @return String representation of this device.
    AM_NODISCARD std::string ToString() const;

    /// @brief Returns the same output as ToString().
    ///
    /// @return String representation of this device.
    AM_NODISCARD std::string str() const;

    /// @brief Creates a device with validation.
    ///
    /// @param type  Device type (CPU, CUDA, CANN).
    /// @param index Device ordinal; -1 means "current device".
    /// @return StatusOr containing the device, or an error if validation fails.
    static StatusOr<Device> Make(DeviceType type, int8_t index = -1);

    /// @brief Parses a device from string format "type[:index]".
    ///
    /// @param text Device string (e.g., "cpu", "cuda:0", "cuda:1").
    /// @return StatusOr containing the device, or an error if parsing fails.
    static StatusOr<Device> FromString(std::string_view text);

    /// @brief Returns the default CPU device (index 0).
    static Device CPU();

    /// @brief Returns the default CUDA device (index -1, "current").
    static Device CUDA();

    /// @brief Returns the default CANN device (index -1, "current").
    static Device CANN();

    bool operator==(const Device& other) const {
        return type() == other.type() && index() == other.index();
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

private:
    DeviceType type_;
    int8_t index_ = -1;

    // Validates invariants; throws on failure.
    void validate() const {
        AM_CHECK(index() >= -1, "Device index must be greater than or equal to -1, but got {}", index());
        AM_CHECK(!is_cpu() || index() <= 0, "CPU device index must be -1 or zero, but got {}", index());
    }
};

/// @brief Converts DeviceType to string representation.
///
/// @param device_type The device type to convert.
/// @param lower_case  If true, returns lowercase (e.g., "cpu"); otherwise "CPU".
/// @return String representation of the device type.
std::string DeviceType2Str(DeviceType device_type, bool lower_case = false);

/// @brief Returns true if device_type is a valid enum value (not kUndefined).
///
/// @param device_type Device type to check.
/// @return True if device_type is not kUndefined.
bool IsValidDeviceType(DeviceType device_type);

/// @brief Returns true if the runtime supports the given device type.
///
/// Currently always returns true for kCPU; GPU types depend on build configuration.
///
/// @param device_type Device type to query.
/// @return True if the device type is supported.
bool IsDeviceKindSupported(DeviceType device_type);

/// @brief Stream output for DeviceType.
std::ostream& operator<<(std::ostream& os, DeviceType device_type);

/// @brief Stream output for Device.
std::ostream& operator<<(std::ostream& os, const Device& device);

}// namespace aethermind

template<>
struct std::hash<aethermind::DeviceType> {
    std::size_t operator()(aethermind::DeviceType k) const noexcept {
        return std::hash<int>()(static_cast<int>(k));
    }
};

template<>
struct std::hash<aethermind::Device> {
    std::size_t operator()(aethermind::Device dev) const noexcept {
        static_assert(sizeof(aethermind::DeviceType) == 1);
        const uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(dev.type())) << 16 |
                              static_cast<uint32_t>(static_cast<uint8_t>(dev.index()));
        return std::hash<uint32_t>()(bits);
    }
};

#endif// AETHERMIND_BASE_DEVICE_H
