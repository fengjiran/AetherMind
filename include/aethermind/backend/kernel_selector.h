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

#endif
