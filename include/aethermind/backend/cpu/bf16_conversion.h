#ifndef AETHERMIND_BACKEND_CPU_BF16_CONVERSION_H
#define AETHERMIND_BACKEND_CPU_BF16_CONVERSION_H

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/dtypes/bf16_conversion.h"

namespace aethermind::cpu {

/// @brief Prepares host conversions using an already-derived feature set.
/// @param effective_features Hardware/OS features restricted by backend policy.
/// @return Allocation-free scalar function table for both directions.
/// @note No CPU detection is performed. ISA-specific implementations may be
///       selected here only after matching the portable numerical contract.
Bf16ConversionFuncs SelectBf16Conversions(
        const CpuFeatureSet& effective_features) noexcept;

} // namespace aethermind::cpu
#endif
