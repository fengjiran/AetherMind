#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H

/// @file alias_utils.h
/// @brief Shared address-range and view-alias checks for CPU kernels.
///
/// Hosts the buffer-aliasing primitives shared by kernels that must reject
/// overlapping input/output storage: half-open byte-address ranges, the
/// overlap predicate over them, and the exact in-place view predicate.
/// Kernels whose alias rules build on these reference them instead of
/// maintaining private copies.

#include "aethermind/base/tensor_view.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Half-open byte-address interval [begin, end).
struct AddressRange {
    std::uintptr_t begin{};
    std::uintptr_t end{};
};

/// @brief Reports whether two half-open address ranges intersect.
inline bool RangesOverlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

/// @brief Reports whether a mutable output is the exact same view as its input.
///
/// True only when base pointer, dtype, rank, extents, and strides all match.
/// Such a view is safe to write in place regardless of other alias rules.
inline bool HasIdenticalMapping(const TensorView& input,
                                const MutableTensorView& output) noexcept {
    if (input.data() != output.data() || input.dtype() != output.dtype() ||
        input.rank() != output.rank()) {
        return false;
    }

    for (int32_t dim = 0; dim < input.rank(); ++dim) {
        if (input.dim(dim) != output.dim(dim) || input.stride(dim) != output.stride(dim)) {
            return false;
        }
    }
    return true;
}

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
