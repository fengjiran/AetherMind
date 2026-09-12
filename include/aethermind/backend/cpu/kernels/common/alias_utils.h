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
struct ByteAddressRange {
    std::uintptr_t begin{};
    std::uintptr_t end{};
};

/// @brief Address geometry for a non-negative-stride two-dimensional view.
///
/// `envelope` and each implicit row range include the bytes between
/// column-strided elements. They can therefore include holes which are not
/// logical elements of the view.
struct RowwiseAddressFootprint {
    ByteAddressRange envelope{};
    std::uintptr_t row_stride_bytes{};
    std::uintptr_t row_envelope_bytes{};
    std::uintptr_t column_stride_bytes{};
    std::uintptr_t element_size_bytes{};
    int64_t row_count{};
    int64_t column_count{};
};

/// @brief Proof result for overlap between two logical byte-address sets.
///
/// `kUnknown` records that address envelopes intersect but stride holes prevent
/// the inexpensive classifier from deciding whether logical element bytes
/// intersect. The same proof states apply to row-wise and general strided
/// footprints.
enum class OverlapClassification : uint8_t {
    kProvenDisjoint,
    kProvenOverlap,
    kUnknown
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
inline bool ByteRangesOverlap(const ByteAddressRange& lhs,
                              const ByteAddressRange& rhs) noexcept {
    if (lhs.begin >= lhs.end || rhs.begin >= rhs.end) {
        return false;
    }
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

/// @brief Builds the exact byte-address range of a contiguous view.
///
/// The range is `[data, data + element_count * element_size)`. An empty view
/// (`element_count == 0`) produces an empty range at `data`; a non-empty view
/// requires non-null data and a non-zero element size. A contiguous view has no
/// stride holes, so intersecting another contiguous range proves a real byte
/// overlap instead of merely suggesting one.
///
/// @param data First byte of the viewed storage.
/// @param element_count Number of logical elements in the view.
/// @param element_size Size of one element in bytes.
/// @param context Caller name used as the error-message prefix.
/// @return The half-open byte range, or InvalidArgument when the geometry is
///         invalid or the address range overflows.
StatusOr<ByteAddressRange> BuildContiguousByteRange(const void* data,
                                                    int64_t element_count,
                                                    size_t element_size,
                                                    std::string_view context) noexcept;

/// @brief Builds checked byte-address geometry for a row-wise view.
///
/// Empty views produce an empty envelope. Non-empty views require non-null
/// data, non-negative extents and strides, and a non-zero element size. The
/// caller supplies `context` solely for neutral diagnostics.
StatusOr<RowwiseAddressFootprint> BuildRowwiseAddressFootprint(
        const void* data,
        int64_t row_count,
        int64_t column_count,
        int64_t row_stride,
        int64_t column_stride,
        size_t element_size,
        std::string_view context) noexcept;

/// @brief Conservatively classifies overlap between two row-wise footprints.
///
/// `kProvenDisjoint` proves the logical element sets are disjoint. `kProvenOverlap`
/// proves overlap when both layouts have no column-stride holes, or when their
/// first elements share an address. Other intersecting row envelopes yield
/// `kUnknown` rather than claiming that holes are logical elements.
OverlapClassification ClassifyRowwiseOverlap(const RowwiseAddressFootprint& lhs,
                                             const RowwiseAddressFootprint& rhs) noexcept;

/// @brief Requires a mutable output and read input to be provably disjoint.
///
/// Maps ClassifyRowwiseOverlap onto the shared status convention: proven
/// overlap is InvalidArgument, while overlap that column-stride holes prevent
/// this helper from deciding is reported as Unimplemented instead of claiming a
/// violation. Callers decide which pairs to validate and grant exemptions such
/// as exact in-place before calling.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param output Address footprint of the mutable output being validated.
/// @param output_role Output role (e.g. "q output") used in the message.
/// @param input Address footprint of the read-only input compared against.
/// @param input_role Input role (e.g. "q") used in the message.
/// @return Ok when the footprints are provably disjoint.
Status ValidateRowwiseDisjoint(std::string_view kernel_name,
                               const RowwiseAddressFootprint& output,
                               std::string_view output_role,
                               const RowwiseAddressFootprint& input,
                               std::string_view input_role) noexcept;

/// @brief Result of classifying whether a layout maps coordinates injectively.
///
/// `kUnknown` records that the inexpensive stride-span proof could not decide.
/// Such a layout may still be injective (shape `[2, 3]` with strides `[3, 2]`
/// is), so it must not be reported as a proven violation.
enum class InjectivityClassification : uint8_t {
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
    ByteAddressRange envelope{};
    int64_t logical_element_count{};
    int64_t max_element_offset{};
    InjectivityClassification injectivity{InjectivityClassification::kUnknown};
    bool is_dense{};
    bool is_empty{};
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
/// @note A violated size or rank precondition is asserted in debug builds and
///       reported as `kUnknown` in release builds instead of overrunning the
///       fixed-rank axis buffer.
InjectivityClassification ClassifyLayoutInjectivity(
        std::span<const int64_t> shape,
        std::span<const int64_t> strides) noexcept;

/// @brief Builds checked address geometry and proven layout properties.
///
/// A view with any zero extent produces an empty footprint whose envelope is
/// `[data, data)`; callers classify empty footprints as disjoint before
/// consulting `is_dense` or `injectivity`. Non-empty views require non-null
/// data, a non-zero element size, and non-negative extents and strides. A rank-0
/// view is a single dense element. Overflowing element counts, offsets, or
/// addresses are invalid geometry rather than an undecidable layout.
StatusOr<StridedAddressFootprint> BuildStridedAddressFootprint(
        const void* data,
        std::span<const int64_t> shape,
        std::span<const int64_t> strides,
        size_t element_size,
        std::string_view context) noexcept;

/// @brief Conservatively classifies overlap between two address footprints.
///
/// Intersecting envelopes are proven overlap when both footprints are dense
/// (their elements fill their envelopes, so a shared byte is a shared element)
/// or when both start at the same address; any other intersection is
/// `kUnknown`.
OverlapClassification ClassifyStridedOverlap(const StridedAddressFootprint& lhs,
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
Status ValidateLayoutInjectivity(std::string_view kernel_name,
                                 InjectivityClassification injectivity,
                                 std::string_view role) noexcept;

/// @brief Requires a mutable output and read input to be provably disjoint.
///
/// General N-D counterpart of ValidateRowwiseDisjoint with the same status
/// convention: proven overlap is InvalidArgument, undecidable envelope
/// intersection is Unimplemented.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param output Footprint of the mutable output being validated.
/// @param output_role Output role (e.g. "output") used in the message.
/// @param input Footprint of the read-only input compared against.
/// @param input_role Input role (e.g. "lhs") used in the message.
/// @return Ok when the footprints are provably disjoint.
Status ValidateStridedDisjoint(std::string_view kernel_name,
                               const StridedAddressFootprint& output,
                               std::string_view output_role,
                               const StridedAddressFootprint& input,
                               std::string_view input_role) noexcept;

/// @brief Reports whether two tensor views have the same logical mapping.
///
/// True only when base pointer, dtype, rank, extents, and strides all match.
/// This is a geometric fact only. The caller decides whether its kernel
/// semantics permit exact in-place execution.
bool HaveIdenticalViewMapping(const TensorView& input,
                              const MutableTensorView& output) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
