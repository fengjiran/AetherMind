#include "aethermind/backend/cpu/bf16_conversion.h"
#include "bf16_conversion_internal.h"

namespace aethermind::cpu {

Bf16ConversionFuncs SelectBf16Conversions(const CpuFeatureSet& effective_features) noexcept {
    Bf16ConversionFuncs funcs;
    if (!effective_features.Contains(CpuFeature::kAvx)) {
        return funcs;
    }

#if defined(BF16_HAS_AVX2_CONVERSION)
    if (effective_features.Contains(CpuFeature::kAvx2)) {
        funcs.to_fp32 = ConvertBf16ToFp32Avx2;
        funcs.from_fp32 = ConvertFp32ToBf16Avx2;
    }
#endif
    return funcs;
}

} // namespace aethermind::cpu
