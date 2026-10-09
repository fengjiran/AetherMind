#ifndef AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H
#define AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H

#include "aethermind/backend/packed_weight.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
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
    /// @brief Packs ordered weight components using the requested exact recipe.
    ///
    /// Composite components must be contiguous rank-2 views sharing one dtype
    /// and feature count. The backend concatenates them along axis 0 in call
    /// order and builds one artifact. A single identity component accepts any
    /// valid rank and must also be contiguous. Tiled layouts require rank-2
    /// float32 components and may add physical padding.
    ///
    /// @param op_type Operator the packed weight serves.
    /// @param components Ordered logical views: one for direct weights, several
    ///        for composites such as Q/K/V or Gate/Up.
    /// @param selector Selector requesting packed CPU weights.
    /// @param recipe Explicit identity or supported tiled layout.
    /// @return Packed artifact, or a validation/allocation error.
    /// @pre Component data must be CPU-accessible. When borrowing from Tensor,
    ///      the caller must validate initialization and CPU device before
    ///      constructing views. Data and metadata must outlive this call.
    AM_NODISCARD StatusOr<std::unique_ptr<PackedWeight>> Pack(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const noexcept;
};

} // namespace aethermind

#endif
