#include "aethermind/backend/cpu/cpu_capabilities.h"

#include <array>
#include <cstddef>
#include <string>

namespace aethermind {
namespace {

constexpr std::array<const char*, static_cast<size_t>(CpuFeature::kCount)> kCpuFeatureNames = {
        "SSE4.1",
        "AVX",
        "AVX2",
        "FMA",
        "F16C",
        "AVX512F",
        "AVX512BW",
        "AVX512VNNI",
        "AVXVNNI",
        "AVX512BF16",
        "AMX_TILE",
        "AMX_INT8",
        "AMX_BF16",
        "NEON",
        "FP16",
        "DOTPROD",
        "I8MM",
        "BF16",
        "SVE",
        "SVE2",
};

} // namespace

const char* ToString(CpuArchitecture architecture) noexcept {
    switch (architecture) {
        case CpuArchitecture::kX86_64:
            return "x86_64";
        case CpuArchitecture::kAArch64:
            return "AArch64";
        case CpuArchitecture::kUnknown:
        default:
            return "Unknown";
    }
}

const char* ToString(CpuFeature feature) noexcept {
    const auto index = static_cast<size_t>(feature);
    if (index >= kCpuFeatureNames.size()) {
        return "Unknown";
    }
    return kCpuFeatureNames[index];
}

std::string ToString(const CpuFeatureSet& features) {
    std::string result{"{"};
    bool first = true;
    for (size_t index = 0; index < static_cast<size_t>(CpuFeature::kCount); ++index) {
        const auto feature = static_cast<CpuFeature>(index);
        if (!features.Contains(feature)) {
            continue;
        }

        if (!first) {
            result += ", ";
        }
        result += ToString(feature);
        first = false;
    }
    result += '}';
    return result;
}

} // namespace aethermind