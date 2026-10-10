#ifndef AETHERMIND_BACKEND_PACKING_RECIPE_H
#define AETHERMIND_BACKEND_PACKING_RECIPE_H

/// @file packing_recipe.h
/// @brief Typed identity of a backend-owned weight-packing layout.

#include "aethermind/base/macros.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace aethermind {

/// @brief Engine-owned identities of exact physical weight layouts.
///
/// Every non-empty identity describes an immutable physical layout. Changes to
/// tile dimensions, panel order or padding require a new identity; backend
/// layout contract headers remain the authority for the physical formulas.
enum class PackingLayout : uint8_t {
    /// No packing layout, or a request awaiting preparation-time resolution.
    kNone = 0,
    kCpuIdentity,
    kCpuBPanelF32Kc512Nr16,
};

/// @brief Checks whether a value identifies a supported, explicit layout.
/// @param layout Layout identity, including potentially invalid enum casts.
/// @return True for a known physical layout; false for kNone or unknown values.
AM_NODISCARD constexpr bool IsValidPackingLayout(PackingLayout layout) noexcept {
    return layout == PackingLayout::kCpuIdentity ||
           layout == PackingLayout::kCpuBPanelF32Kc512Nr16;
}

/// @brief Returns the diagnostic name of a packing layout.
/// @param layout Layout identity to describe.
/// @return Static diagnostic name, or "unknown" for unsupported enum values.
AM_NODISCARD constexpr std::string_view ToString(PackingLayout layout) noexcept {
    switch (layout) {
        case PackingLayout::kNone:
            return "none";
        case PackingLayout::kCpuIdentity:
            return "cpu_identity";
        case PackingLayout::kCpuBPanelF32Kc512Nr16:
            return "cpu_bpanel_f32_kc512_nr16";
    }
    return "unknown";
}

/// @brief Describes the exact packing layout an artifact was produced with.
///
/// The recipe distinguishes layout variants of the same weight binding and
/// selector, and is matched against the consumer descriptor's exact recipe.
struct PackingRecipe {
    PackingLayout layout = PackingLayout::kNone;
    /// Required alignment of layout block starts within the artifact.
    size_t alignment = 0;

    AM_NODISCARD friend bool operator==(const PackingRecipe& lhs,
                                        const PackingRecipe& rhs) = default;
};

} // namespace aethermind

#endif
