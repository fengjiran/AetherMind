#include "aethermind/backend/cpu/bf16_conversion.h"

namespace aethermind::cpu {

Bf16ConversionFuncs SelectBf16Conversions(const CpuFeatureSet& effective_features) noexcept {
    UNUSED(effective_features);
    return {};
}

} // namespace aethermind::cpu
