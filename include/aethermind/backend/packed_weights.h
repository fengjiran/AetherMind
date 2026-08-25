#ifndef AETHERMIND_BACKEND_PACKED_WEIGHTS_H
#define AETHERMIND_BACKEND_PACKED_WEIGHTS_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aethermind {

/// @brief Describes the exact packing layout an artifact was produced with.
///
/// Two artifacts for the same {binding, selector} differ iff their recipes
/// differ; the store keeps them as distinct entries. The recipe is the
/// artifact-side counterpart that a kernel's packed format implies.
struct PackingRecipe {
    /// Canonical layout name (e.g. "cpu_identity").
    std::string layout{};
    /// Required alignment of layout block starts within the artifact.
    size_t alignment = 0;

    AM_NODISCARD friend bool operator==(const PackingRecipe& lhs,
                                        const PackingRecipe& rhs) = default;
};

// Packed weight artifacts are owned by a PackedWeightStore.
// Backend/prepacker code defines the format and build path but does not own
// the packed payload lifetime.
class PackedWeights {
public:
    virtual ~PackedWeights() = default;

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

}// namespace aethermind

#endif
