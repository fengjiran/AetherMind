#ifndef AETHERMIND_BACKEND_PACKED_WEIGHT_H
#define AETHERMIND_BACKEND_PACKED_WEIGHT_H

#include "aethermind/backend/packing_recipe.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aethermind {

/// @brief One opaque backend-layout weight artifact.
///
/// Preparation stores artifacts in a PackedWeightStore and shares their
/// ownership with execution plans. Backend/prepacker code defines the format
/// and build path without retaining ownership of the returned payload.
class PackedWeight {
public:
    virtual ~PackedWeight() = default;

    AM_NODISCARD virtual OpType op_type() const noexcept = 0;
    AM_NODISCARD virtual const KernelSelector& selector() const noexcept = 0;
    AM_NODISCARD virtual const Buffer& storage() const noexcept = 0;
    /// @brief Returns the packing recipe this artifact was produced with.
    AM_NODISCARD virtual const PackingRecipe& recipe() const noexcept = 0;

    /// @brief Logical dtype of the weight this artifact packs.
    AM_NODISCARD virtual DataType logical_dtype() const noexcept = 0;
    /// @brief Logical shape (row-major dims) of the weight this artifact packs.
    AM_NODISCARD virtual const std::vector<int64_t>& logical_shape() const noexcept = 0;
};

} // namespace aethermind

#endif
