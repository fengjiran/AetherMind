#ifndef AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H
#define AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H

#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

namespace aethermind {

/// @brief CPU identity-packing recipe constants.
///
/// The Phase 1 CPU prepacker copies logical Float32 weights without a layout
/// transformation. Producers attach this recipe to the artifact and packed
/// kernel consumers validate the same recipe before interpreting its storage.
/// Keeping these constants beside the prepacker prevents the two sides from
/// drifting when tiled or blocked CPU packing recipes are added later.
namespace cpu {
inline constexpr std::string_view kCpuIdentityPackingLayout = "cpu_identity";
inline constexpr size_t kCpuIdentityPackingAlignment = 64;
} // namespace cpu

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

    /// @brief Packs recipe-ordered weight components into one fused artifact.
    ///
    /// The backend owns the layout authority: components must be valid
    /// contiguous rank-2 views sharing one dtype and feature count. They are
    /// concatenated along axis 0 in call order (Q/K/V or Gate/Up) into an
    /// aligned buffer sized as the sum of their logical bytes.
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

    /// @brief Returns the packing recipe this prepacker produces.
    ///
    /// Deterministic per selector; the packed artifact carries the same recipe
    /// so a PackedWeightStore can verify key/artifact consistency.
    AM_NODISCARD static PackingRecipe RecipeFor(
            const KernelSelector& selector) noexcept;
};

} // namespace aethermind

#endif
