#ifndef AETHERMIND_BACKEND_CPU_CPU_BACKEND_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_CPU_BACKEND_INTERNAL_H

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/base/kernel_selector.h"

namespace aethermind::cpu::internal {

/// @brief Shared eligibility/priority resolver used by recipe and kernel queries.
StatusOr<const KernelDescriptor*> ResolveEligibleDescriptor(
        const KernelRegistry& registry,
        OpType op_type,
        const KernelSelector& selector,
        const CpuFeatureSet& effective_features);

/// @brief Resolves the same descriptor and extracts its exact packed recipe.
StatusOr<PackingRecipe> ResolvePackingRecipeFromRegistry(
        const KernelRegistry& registry,
        OpType op_type,
        const KernelSelector& selector,
        const CpuFeatureSet& effective_features);

} // namespace aethermind::cpu::internal

#endif // AETHERMIND_BACKEND_CPU_CPU_BACKEND_INTERNAL_H
