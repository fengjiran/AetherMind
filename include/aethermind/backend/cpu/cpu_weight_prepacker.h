#ifndef AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H
#define AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H

#include "aethermind/backend/cpu/cpu_bpanel_packing.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief CPU identity-packing recipe constants.
///
/// The CPU identity recipe copies logical weights without a layout
/// transformation. Producers attach this recipe to the artifact and packed
/// kernel consumers validate it before interpreting storage. Tiled layouts
/// use their own versioned recipes in cpu_bpanel_packing.h.
namespace cpu {
inline constexpr std::string_view kCpuIdentityPackingLayout = "cpu_identity";
inline constexpr size_t kCpuIdentityPackingAlignment = 64;
} // namespace cpu

inline PackingRecipe CpuIdentityPackingRecipe() {
    return PackingRecipe{.layout = cpu::kCpuIdentityPackingLayout,
                         .alignment = cpu::kCpuIdentityPackingAlignment};
}

class CpuWeightPrepacker {
public:
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            const Tensor& logical_weight,
            const KernelSelector& selector) const noexcept;

    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            TensorView logical_weight,
            const KernelSelector& selector) const noexcept;

    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            TensorView logical_weight,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const noexcept;

    /// @brief Packs recipe-ordered weight components into one fused artifact.
    ///
    /// The backend owns the layout authority: components must be valid
    /// contiguous rank-2 views sharing one dtype and feature count. They are
    /// concatenated along axis 0 in call order (Q/K/V or Gate/Up), then packed
    /// according to the explicit recipe. The identity layout uses the logical
    /// byte size; tiled layouts may add padding.
    ///
    /// @param op_type Operator the packed weight serves.
    /// @param components Recipe-ordered logical weight views to pack; exactly
    ///        one for direct bindings.
    /// @param selector Selector requesting `WeightFormat::kPacked` on CPU.
    /// @return Fused packed artifact, or an error describing the first
    ///         violation.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector) const noexcept;

    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeights>> Pack(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const noexcept;

    /// @brief Returns the legacy CPU identity recipe.
    ///
    /// Production packing receives its exact recipe from the selected kernel
    /// descriptor and passes it explicitly to Pack.
    AM_NODISCARD static PackingRecipe RecipeFor(
            const KernelSelector& selector) noexcept;
};

} // namespace aethermind

#endif
