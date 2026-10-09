#ifndef AETHERMIND_BACKEND_CPU_CPU_IDENTITY_PACKING_H
#define AETHERMIND_BACKEND_CPU_CPU_IDENTITY_PACKING_H

/// @file cpu_identity_packing.h
/// @brief Shared CPU identity-layout contract for packers and kernel consumers.

#include "aethermind/backend/packing_recipe.h"

#include <cstddef>

namespace aethermind {

/// @brief CPU identity-packing recipe constants.
///
/// The CPU identity recipe copies logical weights without a layout
/// transformation. Producers attach this recipe to the artifact and packed
/// kernel consumers validate it before interpreting storage. Tiled layouts
/// use their own versioned recipes in cpu_bpanel_packing.h.
namespace cpu {
inline constexpr PackingLayout kCpuIdentityPackingLayout = PackingLayout::kCpuIdentity;
inline constexpr size_t kCpuIdentityPackingAlignment = 64;
} // namespace cpu

/// @brief Returns the canonical CPU identity-packing recipe.
/// @return Identity layout name and required alignment.
inline PackingRecipe CpuIdentityPackingRecipe() {
    return {.layout = cpu::kCpuIdentityPackingLayout,
            .alignment = cpu::kCpuIdentityPackingAlignment};
}

} // namespace aethermind

#endif
