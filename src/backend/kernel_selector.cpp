#include "aethermind/backend/kernel_selector.h"

#include <string>

namespace aethermind {

const char* ToString(IsaLevel isa) noexcept {
    switch (isa) {
        case IsaLevel::kScalar:
            return "Scalar";
        case IsaLevel::kAVX2:
            return "AVX2";
        case IsaLevel::kAVX512:
            return "AVX512";
        case IsaLevel::kAMX:
            return "AMX";
        default:
            return "Unknown";
    }
}

const char* ToString(ExecPhase phase) noexcept {
    switch (phase) {
        case ExecPhase::kPrefill:
            return "Prefill";
        case ExecPhase::kDecode:
            return "Decode";
        case ExecPhase::kBoth:
            return "Both";
        default:
            return "Unknown";
    }
}

const char* ToString(WeightFormat format) noexcept {
    switch (format) {
        case WeightFormat::kPlain:
            return "Plain";
        case WeightFormat::kPacked:
            return "Packed";
        case WeightFormat::kQuantizedInt8:
            return "QuantizedInt8";
        case WeightFormat::kQuantizedInt4:
            return "QuantizedInt4";
        default:
            return "Unknown";
    }
}

std::string ToString(const KernelSelector& selector) {
    return std::string("KernelSelector{device=") +
           DeviceType2Str(selector.device_type) +
           ", activation_dtype=" +
           std::to_string(selector.activation_dtype.bits()) +
           "bit" +
           ", weight_dtype=" +
           std::to_string(selector.weight_dtype.bits()) +
           "bit" +
           ", weight_format=" +
           ToString(selector.weight_format) +
           ", isa=" +
           ToString(selector.isa) +
           ", phase=" +
           ToString(selector.phase) +
           "}";
}

}// namespace aethermind