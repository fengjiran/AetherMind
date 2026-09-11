#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H

/// @file alias_utils.h
/// @brief Shared address-range and view-alias checks for CPU kernels.
///
/// Hosts the buffer-aliasing primitives shared by kernels that must reject
/// overlapping input/output storage: half-open byte-address ranges, exact
/// in-place view checks, and conservative row-wise layout classification.
/// Kernels keep their own alias policies and use these geometry primitives
/// instead of maintaining private copies.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Half-open byte-address interval [begin, end).
struct AddressRange {
    std::uintptr_t begin{};
    std::uintptr_t end{};
};

/// @brief Address geometry for a non-negative-stride two-dimensional view.
///
/// `envelope` and each implicit row range include the bytes between
/// column-strided elements. They can therefore include holes which are not
/// logical elements of the view.
struct RowwiseAddressLayout {
    AddressRange envelope{};
    std::uintptr_t row_stride_bytes{};
    std::uintptr_t row_envelope_bytes{};
    std::uintptr_t column_stride_bytes{};
    std::uintptr_t item_size_bytes{};
    int64_t row_count{};
    int64_t column_count{};
};

/// @brief Result of conservatively classifying two row-wise layouts.
///
/// `kMayOverlap` is not proof of logical-element overlap. It records that the
/// row envelopes intersect but column-stride holes prevent this helper from
/// deciding whether the logical element sets intersect.
enum class RowwiseLayoutOverlap : uint8_t {
    kDisjoint,
    kProvenOverlap,
    kMayOverlap
};

/// @brief Reports whether two half-open address ranges intersect.
///
/// Only a shared element counts as an intersection: empty ranges
/// (`begin == end`) never overlap, even when their address coincides with a
/// point inside the other range. Ranges are required to be well-formed
/// (`begin <= end`); a reversed or wrapped range is left undefined.
///
/// @param lhs First half-open byte interval `[begin, end)`.
/// @param rhs Second half-open byte interval `[begin, end)`.
/// @return True when the two ranges share at least one byte address.
inline bool RangesOverlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
    if (lhs.begin >= lhs.end || rhs.begin >= rhs.end) {
        return false;
    }
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

/// @brief Builds checked byte-address geometry for a row-wise view.
///
/// Empty views produce an empty envelope. Non-empty views require non-null
/// data, non-negative extents and strides, and a non-zero item size. The
/// caller supplies `context` solely for neutral diagnostics.
StatusOr<RowwiseAddressLayout> BuildRowwiseAddressLayout(
        const void* data,
        int64_t row_count,
        int64_t column_count,
        int64_t row_stride,
        int64_t column_stride,
        size_t item_size,
        std::string_view context) noexcept;

/// @brief Conservatively classifies overlap between two row-wise layouts.
///
/// `kDisjoint` proves the logical element sets are disjoint. `kProvenOverlap`
/// proves overlap when both layouts have no column-stride holes, or when their
/// first elements share an address. Other intersecting row envelopes yield
/// `kMayOverlap` rather than claiming that holes are logical elements.
RowwiseLayoutOverlap ClassifyRowwiseLayoutOverlap(const RowwiseAddressLayout& lhs,
                                                  const RowwiseAddressLayout& rhs) noexcept;

/// @brief Reports whether row-wise logical views may overlap.
///
/// A false result proves disjointness. A true result includes both proven
/// overlap and cases where column-stride holes prevent an exact conclusion.
bool RowwiseLayoutsMayOverlap(const RowwiseAddressLayout& lhs,
                              const RowwiseAddressLayout& rhs) noexcept;

/// @brief Reports whether a mutable output is the exact same view as its input.
///
/// True only when base pointer, dtype, rank, extents, and strides all match.
/// Such a view is safe to write in place regardless of other alias rules.
bool HasIdenticalMapping(const TensorView& input,
                         const MutableTensorView& output) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
