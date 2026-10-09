#ifndef AETHERMIND_BACKEND_CPU_CPU_BPANEL_PACKING_H
#define AETHERMIND_BACKEND_CPU_CPU_BPANEL_PACKING_H

/// @file cpu_bpanel_packing.h
/// @brief Versioned CPU FP32 GEMM packed-B layout contract.

#include "aethermind/backend/packing_recipe.h"
#include "aethermind/base/status.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::cpu {

inline constexpr PackingLayout kCpuBPanelF32V1Avx2Layout = PackingLayout::kCpuBPanelF32V1Avx2;
inline constexpr size_t kCpuBPanelF32V1Alignment = 64;
inline constexpr int64_t kCpuBPanelF32V1NR = 16;
inline constexpr int64_t kCpuBPanelF32V1KC = 512;

/// @brief Returns the recipe identifying the v1 AVX2 FP32 B-panel candidate.
inline PackingRecipe CpuBPanelF32V1Avx2Recipe() {
    return {.layout = kCpuBPanelF32V1Avx2Layout,
            .alignment = kCpuBPanelF32V1Alignment};
}

/// @brief Computes exact artifact bytes for the padded v1 B-panel layout.
StatusOr<size_t> CpuBPanelF32V1PackedByteSize(int64_t n, int64_t k) noexcept;

} // namespace aethermind::cpu

#endif // AETHERMIND_BACKEND_CPU_CPU_BPANEL_PACKING_H
