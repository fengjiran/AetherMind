#ifndef AETHERMIND_BACKEND_CPU_CPU_PACKED_WEIGHT_LAYOUT_H
#define AETHERMIND_BACKEND_CPU_CPU_PACKED_WEIGHT_LAYOUT_H

/// @file cpu_packed_weight_layout.h
/// @brief Shared immutable CPU packed-weight formats for producers and consumers.
///
/// Layout identities define physical bytes independently of kernel registry,
/// capabilities and consumer ISA eligibility. Changing tile dimensions, axis
/// order or padding requires a distinct PackingLayout identity.

#include "aethermind/backend/packing_recipe.h"
#include "aethermind/base/status.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {
namespace cpu {

/// Identity stores contiguous logical bytes without a layout transformation.
/// Producers attach this recipe to artifacts; consumers validate it before
/// interpreting storage.
inline constexpr PackingLayout kCpuIdentityPackingLayout = PackingLayout::kCpuIdentity;
inline constexpr size_t kCpuIdentityPackingAlignment = 64;

/// FP32 logical [N,K] weights use the physical axis order
/// [K-panel][N-block][K-row][N-lane], with shape [ceil(K/512), ceil(N/16), 512, 16].
/// N lanes are contiguous; padding beyond logical K or N contains +0.0F.
/// Each panel block starts at 64-byte alignment.
inline constexpr auto kCpuBPanelF32Kc512Nr16Layout = PackingLayout::kCpuBPanelF32Kc512Nr16;
inline constexpr size_t kCpuBPanelF32Kc512Nr16Alignment = 64;
inline constexpr int64_t kCpuBPanelF32Kc512Nr16NR = 16;
inline constexpr int64_t kCpuBPanelF32Kc512Nr16KC = 512;

/// @brief Returns the recipe for the KC=512, NR=16 FP32 B-panel layout.
inline PackingRecipe CpuBPanelF32Kc512Nr16Recipe() {
    return {.layout = kCpuBPanelF32Kc512Nr16Layout,
            .alignment = kCpuBPanelF32Kc512Nr16Alignment};
}

/// @brief Computes exact artifact bytes for the padded KC=512, NR=16 B-panel layout.
StatusOr<size_t> CpuBPanelF32Kc512Nr16PackedByteSize(int64_t n, int64_t k) noexcept;

} // namespace cpu

/// @brief Returns the canonical CPU identity-packing recipe.
/// @return Identity layout and required alignment.
inline PackingRecipe CpuIdentityPackingRecipe() {
    return {.layout = cpu::kCpuIdentityPackingLayout,
            .alignment = cpu::kCpuIdentityPackingAlignment};
}

} // namespace aethermind

#endif // AETHERMIND_BACKEND_CPU_CPU_PACKED_WEIGHT_LAYOUT_H
