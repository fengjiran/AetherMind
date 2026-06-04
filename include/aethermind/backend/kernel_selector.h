#ifndef AETHERMIND_BACKEND_KERNEL_SELECTOR_H
#define AETHERMIND_BACKEND_KERNEL_SELECTOR_H

#include "data_type.h"
#include "device.h"

#include <cstdint>

namespace aethermind {

enum class IsaLevel : uint8_t {
    kScalar = 0,
    kAVX2,
    kAVX512,
    kAMX,
};

enum class ExecPhase : uint8_t {
    kPrefill = 0,
    kDecode,
    kBoth,
};

enum class WeightFormat : uint8_t {
    kPlain = 0,
    kPacked,
    kQuantizedInt8,
    kQuantizedInt4,
};

AM_NODISCARD const char* ToString(IsaLevel isa) noexcept;
AM_NODISCARD const char* ToString(ExecPhase phase) noexcept;
AM_NODISCARD const char* ToString(WeightFormat format) noexcept;

AM_NODISCARD inline bool PhaseMatches(ExecPhase candidate,
                                      ExecPhase request) noexcept {
    return candidate == request || candidate == ExecPhase::kBoth;
}

struct KernelSelector {
    DeviceType device_type = DeviceType::kUndefined;
    DataType activation_dtype{};
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;

    friend bool operator==(const KernelSelector& lhs, const KernelSelector& rhs) noexcept {
        return lhs.device_type == rhs.device_type &&
               lhs.activation_dtype == rhs.activation_dtype &&
               lhs.weight_dtype == rhs.weight_dtype &&
               lhs.weight_format == rhs.weight_format &&
               lhs.isa == rhs.isa &&
               lhs.phase == rhs.phase;
    }

    friend bool operator!=(const KernelSelector& lhs, const KernelSelector& rhs) {
        return !(lhs == rhs);
    }
};

AM_NODISCARD inline bool SelectorMatches(const KernelSelector& candidate,
                                         const KernelSelector& request) noexcept {
    return candidate.device_type == request.device_type &&
           candidate.activation_dtype == request.activation_dtype &&
           candidate.weight_dtype == request.weight_dtype &&
           candidate.weight_format == request.weight_format &&
           PhaseMatches(candidate.phase, request.phase) &&
           candidate.isa <= request.isa;
}

std::string ToString(const KernelSelector& selector);

}// namespace aethermind

template<>
struct std::hash<aethermind::KernelSelector> {
    std::size_t operator()(const aethermind::KernelSelector& s) const noexcept {
        // Pack small fields into a single size_t on 64-bit platforms.
        // device_type(8) | isa(8) | phase(8) | weight_format(8) = 32 bits
        const auto lo = static_cast<std::size_t>(s.device_type) |
                        (static_cast<std::size_t>(s.isa) << 8) |
                        (static_cast<std::size_t>(s.phase) << 16) |
                        (static_cast<std::size_t>(s.weight_format) << 24);
        const auto hi = std::hash<aethermind::DataType>{}(s.activation_dtype) ^
                        (std::hash<aethermind::DataType>{}(s.weight_dtype) << 1);
        return lo ^ (hi << 32);
    }
};

#endif
