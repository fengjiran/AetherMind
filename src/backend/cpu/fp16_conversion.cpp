#include "aethermind/backend/cpu/fp16_conversion.h"
#include "fp16_conversion_internal.h"

namespace aethermind::cpu {

Fp16ConversionFuncs SelectFp16Conversions(const CpuFeatureSet& effective_features) noexcept {
    Fp16ConversionFuncs funcs;
    if (!effective_features.Contains(CpuFeature::kAvx)) {
        return funcs;
    }

#if defined(FP16_HAS_F16C_CONVERSION)
    if (effective_features.Contains(CpuFeature::kF16c)) {
        funcs.to_fp32 = ConvertFp16ToFp32F16c;
    }
#endif

#if defined(FP16_HAS_AVX2_CONVERSION)
    if (effective_features.Contains(CpuFeature::kAvx2)) {
        funcs.from_fp32 = ConvertFp32ToFp16Avx2;
    }
#endif
    return funcs;
}

} // namespace aethermind::cpu
