#ifndef AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H
#define AETHERMIND_BACKEND_CPU_CPU_WEIGHT_PREPACKER_H

#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"

#include <memory>

namespace aethermind {

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

    /// @brief Returns the packing recipe this prepacker produces.
    ///
    /// Deterministic per selector; the packed artifact carries the same recipe
    /// so a PackedWeightStore can verify key/artifact consistency.
    AM_NODISCARD static PackingRecipe RecipeFor(
            const KernelSelector& selector) noexcept;
};

} // namespace aethermind

#endif
