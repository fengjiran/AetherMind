#ifndef AETHERMIND_BACKEND_CPU_FP16_CONVERSION_H
#define AETHERMIND_BACKEND_CPU_FP16_CONVERSION_H

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/dtypes/fp16_conversion.h"

namespace aethermind::cpu {

/// @brief Selects compiled host conversions using an already-derived feature set.
/// @param effective_features Hardware/OS features restricted by backend policy.
/// @return Allocation-free function table with a scalar fallback per direction.
/// @note Does not probe CPU capabilities. Callers must supply actual effective
///       features, not unsupported synthetic features, before invoking the table.
Fp16ConversionFuncs SelectFp16Conversions(
        const CpuFeatureSet& effective_features) noexcept;

} // namespace aethermind::cpu
#endif
