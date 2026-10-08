#ifndef AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H
#define AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H

#include "aethermind/backend/packed_weight.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"

#include <memory>
#include <span>

namespace aethermind {

/// @brief CPU weight-packing service for explicit backend layout recipes.
///
/// Prepacker describes model-preparation timing; the recipe passed to Pack
/// determines the packed artifact layout.
class CpuWeightPrepacker {
public:
    /// @brief Compatibility overload using the CPU identity recipe.
    /// @param op_type Operator the packed weight serves.
    /// @param logical_weight Logical CPU tensor to pack.
    /// @param selector Selector requesting packed CPU weights.
    /// @return Identity-packed artifact, or a validation/allocation error.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeight>> Pack(
            OpType op_type,
            const Tensor& logical_weight,
            const KernelSelector& selector) const noexcept;

    /// @brief Compatibility overload using the CPU identity recipe.
    /// @param op_type Operator the packed weight serves.
    /// @param logical_weight Contiguous logical weight view to pack.
    /// @param selector Selector requesting packed CPU weights.
    /// @return Identity-packed artifact, or a validation/allocation error.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeight>> Pack(
            OpType op_type,
            TensorView logical_weight,
            const KernelSelector& selector) const noexcept;

    /// @brief Packs one logical weight using the requested exact recipe.
    /// @param op_type Operator the packed weight serves.
    /// @param logical_weight Contiguous logical weight view to pack.
    /// @param selector Selector requesting packed CPU weights.
    /// @param recipe Explicit identity or supported tiled layout.
    /// @return Packed artifact, or a validation/allocation error.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeight>> Pack(
            OpType op_type,
            TensorView logical_weight,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const noexcept;

    /// @brief Compatibility overload using the CPU identity recipe.
    /// @param op_type Operator the packed weight serves.
    /// @param components Recipe-ordered logical weight views to pack.
    /// @param selector Selector requesting packed CPU weights.
    /// @return Identity-packed artifact, or a validation/allocation error.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeight>> Pack(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector) const noexcept;

    /// @brief Packs ordered weight components using the requested exact recipe.
    ///
    /// Composite components must be contiguous rank-2 views sharing one dtype
    /// and feature count. The backend concatenates them along axis 0 in call
    /// order and builds one artifact. A single component follows the
    /// single-view contract. Tiled layouts may add physical padding.
    ///
    /// @param op_type Operator the packed weight serves.
    /// @param components Ordered logical views: one for direct weights, several
    ///        for composites such as Q/K/V or Gate/Up.
    /// @param selector Selector requesting packed CPU weights.
    /// @param recipe Explicit identity or supported tiled layout.
    /// @return Packed artifact, or a validation/allocation error.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeight>> Pack(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const noexcept;

    /// @brief Compatibility query returning the fixed CPU identity recipe.
    ///
    /// Production packing receives its exact recipe from the selected kernel
    /// descriptor and passes it explicitly to Pack.
    ///
    /// @param selector Retained for compatibility; ignored by this query.
    /// @return Canonical CPU identity recipe.
    AM_NODISCARD static PackingRecipe RecipeFor(
            const KernelSelector& selector) noexcept;
};

} // namespace aethermind

#endif
