#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H

/// @file alias_utils.h
/// @brief Shared address-range and view-alias checks for CPU kernels.
///
/// Hosts the buffer-aliasing primitives shared by kernels that must reject
/// overlapping input/output storage: half-open byte-address ranges, exact
/// in-place view checks, conservative row-wise layout classification, and the
/// status mapping that turns a classification into a diagnostic. Kernels own
/// which tensor pairs they validate and which exemptions they grant (such as
/// exact in-place); they use these primitives instead of maintaining private
/// copies.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

#include <cstddef>
#include <cstdint>
#include <span>
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
    std::uintptr_t col_stride_bytes{};
    std::uintptr_t item_size_bytes{};
    int64_t row_count{};
    int64_t col_count{};
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

/// @brief Builds the exact byte-address range of a contiguous view.
///
/// The range is `[data, data + element_count * item_size)`. An empty view
/// (`element_count == 0`) produces an empty range at `data`; a non-empty view
/// requires non-null data and a non-zero item size. A contiguous view has no
/// stride holes, so intersecting another contiguous range proves a real byte
/// overlap instead of merely suggesting one.
///
/// @param data First byte of the viewed storage.
/// @param element_count Number of logical elements in the view.
/// @param item_size Size of one element in bytes.
/// @param context Caller name used as the error-message prefix.
/// @return The half-open byte range, or InvalidArgument when the geometry is
///         invalid or the address range overflows.
StatusOr<AddressRange> BuildContiguousAddressRange(const void* data,
                                                   int64_t element_count,
                                                   size_t item_size,
                                                   std::string_view context) noexcept;

/// @brief Builds checked byte-address geometry for a row-wise view.
///
/// Empty views produce an empty envelope. Non-empty views require non-null
/// data, non-negative extents and strides, and a non-zero item size. The
/// caller supplies `context` solely for neutral diagnostics.
StatusOr<RowwiseAddressLayout> BuildRowwiseAddressLayout(
        const void* data,
        int64_t row_count,
        int64_t col_count,
        int64_t row_stride,
        int64_t col_stride,
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

/// @brief Rejects a mutable output whose row-wise layout overlaps a read input.
///
/// Maps ClassifyRowwiseLayoutOverlap onto the shared status convention: proven
/// overlap is InvalidArgument, while overlap that column-stride holes prevent
/// this helper from deciding is reported as Unimplemented instead of claiming a
/// violation. Callers decide which pairs to validate and grant exemptions such
/// as exact in-place before calling.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param output Layout of the mutable output being validated.
/// @param output_role Output role (e.g. "q output") used in the message.
/// @param input Layout of the read-only input compared against.
/// @param input_role Input role (e.g. "q") used in the message.
/// @return Ok when the layouts are provably disjoint.
Status ValidateNoRowwiseOverlap(std::string_view kernel_name,
                                const RowwiseAddressLayout& output,
                                std::string_view output_role,
                                const RowwiseAddressLayout& input,
                                std::string_view input_role) noexcept;

/// @brief Result of classifying whether a layout maps coordinates injectively.
///
/// `kUnknown` records that the inexpensive stride-span proof could not decide.
/// Such a layout may still be injective (shape `[2, 3]` with strides `[3, 2]`
/// is), so it must not be reported as a proven violation.
enum class LayoutInjectivity : uint8_t {
    kProvenInjective,
    kProvenNonInjective,
    kUnknown
};

/// @brief Address geometry and proven layout properties of a strided view.
///
/// `envelope` spans from the base address past the last logical element and can
/// contain holes that are not logical elements of the view. `is_dense` reports
/// that the logical elements exactly fill the envelope without holes or
/// repeats; that is weaker than row-major contiguity, since a transposed layout
/// can be dense without being contiguous.
struct StridedAddressFootprint {
    AddressRange envelope{};
    int64_t numel{};
    int64_t max_offset{};
    LayoutInjectivity injectivity{LayoutInjectivity::kUnknown};
    bool is_dense{};
    bool is_empty{};
};

/// @brief Result of conservatively classifying two address footprints.
///
/// `kMayOverlap` is not proof of overlap: it records that the envelopes
/// intersect while stride holes prevent deciding whether the logical elements
/// do.
enum class FootprintOverlap : uint8_t {
    kDisjoint,
    kProvenOverlap,
    kMayOverlap
};

/// @brief Classifies whether `(shape, strides)` maps coordinates to distinct offsets.
///
/// Axes with extent > 1 are visited in ascending stride order while tracking
/// the span covered so far and whether the lower axes fill that span without
/// holes. A stride below the covered span collides with a reachable offset when
/// the lower axes are dense, and is undecidable otherwise.
///
/// @param shape Extents per axis.
/// @param strides Strides in elements per axis.
/// @return The proof result; arithmetic overflow yields `kUnknown`, while
///         BuildStridedAddressFootprint reports it as invalid geometry.
/// @pre `shape.size() == strides.size()`, `shape.size() <= ShapeAndStride::kMaxRank`,
///      and every stride is non-negative.
LayoutInjectivity ClassifyLayoutInjectivity(std::span<const int64_t> shape,
                                            std::span<const int64_t> strides) noexcept;

/// @brief Builds checked address geometry and proven layout properties.
///
/// A view with any zero extent produces an empty footprint whose envelope is
/// `[data, data)`; callers classify empty footprints as disjoint before
/// consulting `is_dense` or `injectivity`. Non-empty views require non-null
/// data, a non-zero item size, and non-negative extents and strides. A rank-0
/// view is a single dense element. Overflowing element counts, offsets, or
/// addresses are invalid geometry rather than an undecidable layout.
StatusOr<StridedAddressFootprint> BuildStridedAddressFootprint(
        const void* data,
        std::span<const int64_t> shape,
        std::span<const int64_t> strides,
        size_t item_size,
        std::string_view context) noexcept;

/// @brief Conservatively classifies overlap between two address footprints.
///
/// Intersecting envelopes are proven overlap when both footprints are dense
/// (their elements fill their envelopes, so a shared byte is a shared element)
/// or when both start at the same address; any other intersection is
/// `kMayOverlap`.
FootprintOverlap ClassifyFootprintOverlap(const StridedAddressFootprint& lhs,
                                          const StridedAddressFootprint& rhs) noexcept;

/// @brief Maps a layout injectivity classification onto the shared status convention.
///
/// A proven collision is InvalidArgument; a layout the inexpensive proof cannot
/// decide is Unimplemented rather than a claimed violation.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param injectivity Classification of the layout being validated.
/// @param role Tensor role (e.g. "output") used in the message.
/// @return Ok when the layout is provably injective.
Status ValidateInjectiveLayout(std::string_view kernel_name,
                               LayoutInjectivity injectivity,
                               std::string_view role) noexcept;

/// @brief Rejects a mutable output whose footprint overlaps a read input.
///
/// General N-D counterpart of ValidateNoRowwiseOverlap with the same status
/// convention: proven overlap is InvalidArgument, undecidable envelope
/// intersection is Unimplemented.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param output Footprint of the mutable output being validated.
/// @param output_role Output role (e.g. "output") used in the message.
/// @param input Footprint of the read-only input compared against.
/// @param input_role Input role (e.g. "lhs") used in the message.
/// @return Ok when the footprints are provably disjoint.
Status ValidateNoFootprintOverlap(std::string_view kernel_name,
                                  const StridedAddressFootprint& output,
                                  std::string_view output_role,
                                  const StridedAddressFootprint& input,
                                  std::string_view input_role) noexcept;

/// @brief Reports whether a mutable output is the exact same view as its input.
///
/// True only when base pointer, dtype, rank, extents, and strides all match.
/// Such a view is safe to write in place regardless of other alias rules.
bool HasIdenticalMapping(const TensorView& input,
                         const MutableTensorView& output) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
