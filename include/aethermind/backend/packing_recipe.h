#ifndef AETHERMIND_BACKEND_PACKING_RECIPE_H
#define AETHERMIND_BACKEND_PACKING_RECIPE_H

/// @file packing_recipe.h
/// @brief Lightweight identity of a backend-owned weight-packing layout.

#include "aethermind/base/macros.h"

#include <cstddef>
#include <string_view>

namespace aethermind {

/// @brief Describes the exact packing layout an artifact was produced with.
///
/// Two artifacts for the same {binding, selector} differ iff their recipes
/// differ; the store keeps them as distinct entries. The recipe is the
/// artifact-side counterpart that a kernel's packed format implies.
struct PackingRecipe {
    /// Canonical layout name (e.g. "cpu_identity"). Borrows storage that must
    /// outlive every artifact and artifact key carrying this recipe, so it must
    /// reference a string literal or a static-duration constant such as
    /// `cpu::kCpuIdentityPackingLayout`.
    std::string_view layout{};
    /// Required alignment of layout block starts within the artifact.
    size_t alignment = 0;

    AM_NODISCARD friend bool operator==(const PackingRecipe& lhs,
                                        const PackingRecipe& rhs) = default;
};

} // namespace aethermind

#endif
