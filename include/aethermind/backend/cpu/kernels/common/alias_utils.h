#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H

/// @file alias_utils.h
/// @brief Shared address-range and view-alias checks for CPU kernels.
///
/// Hosts the buffer-aliasing primitives shared by kernels that must reject
/// overlapping input/output storage: half-open byte-address ranges, exact
/// in-place view checks, complete row-wise view analysis, and the status
/// mapping that turns a classification into a diagnostic. Kernels own which
/// tensor pairs they validate and which exemptions they grant (such as exact
/// in-place); they use these primitives instead of maintaining private copies.

#include "aethermind/base/macros.h"
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

/// @brief Address geometry for a non-negative-stride two-dimensional view.
///
/// `envelope` and each implicit row range include the bytes between
/// column-strided elements. They can therefore include holes which are not
/// logical elements of the view. Instances are produced only by the checked
/// builders in this component and expose read-only facts to callers.
class RowwiseAddressFootprint {
public:
    AM_NODISCARD ByteAddressRange envelope() const noexcept {
        return envelope_;
    }

    AM_NODISCARD std::uintptr_t row_stride_bytes() const noexcept {
        return row_stride_bytes_;
    }

    AM_NODISCARD std::uintptr_t row_envelope_bytes() const noexcept {
        return row_envelope_bytes_;
    }

    AM_NODISCARD std::uintptr_t column_stride_bytes() const noexcept {
        return column_stride_bytes_;
    }

    AM_NODISCARD std::uintptr_t element_size_bytes() const noexcept {
        return element_size_bytes_;
    }

    AM_NODISCARD int64_t row_count() const noexcept {
        return row_count_;
    }

    AM_NODISCARD int64_t column_count() const noexcept {
        return column_count_;
    }

    AM_NODISCARD bool is_empty() const noexcept {
        return is_empty_;
    }

private:
    RowwiseAddressFootprint(ByteAddressRange envelope,
                            std::uintptr_t row_stride_bytes,
                            std::uintptr_t row_envelope_bytes,
                            std::uintptr_t column_stride_bytes,
                            std::uintptr_t element_size_bytes,
                            int64_t row_count,
                            int64_t column_count,
                            bool is_empty) noexcept;

    friend class AliasUtilsFactory;

    ByteAddressRange envelope_{};
    std::uintptr_t row_stride_bytes_{};
    std::uintptr_t row_envelope_bytes_{};
    std::uintptr_t column_stride_bytes_{};
    std::uintptr_t element_size_bytes_{};
    int64_t row_count_{};
    int64_t column_count_{};
    bool is_empty_{};
};

/// @brief Checked row-wise geometry of a tensor view.
///
/// The analysis flattens a normal tensor view into `[row_count, column_count]`.
/// Its `footprint` is safe to pass to the alias validators. The analysis
/// records facts only: read-only inputs may have overlapping row envelopes,
/// while ValidateRowwiseOutputLayout applies the stricter mutable-output
/// policy. Callers cannot construct or alter an analysis piecemeal.
class RowwiseViewAnalysis {
public:
    AM_NODISCARD const RowwiseAddressFootprint& footprint() const& noexcept {
        return footprint_;
    }

    AM_NODISCARD const RowwiseAddressFootprint& footprint() const&& = delete;

    AM_NODISCARD int64_t row_count() const noexcept {
        return footprint_.row_count();
    }

    AM_NODISCARD int64_t column_count() const noexcept {
        return footprint_.column_count();
    }

    AM_NODISCARD int64_t row_stride() const noexcept {
        return row_stride_;
    }

    AM_NODISCARD int64_t column_stride() const noexcept {
        return column_stride_;
    }

    AM_NODISCARD int64_t max_element_offset() const noexcept {
        return max_element_offset_;
    }

    AM_NODISCARD InjectivityClassification injectivity() const noexcept {
        return injectivity_;
    }

    AM_NODISCARD bool has_disjoint_row_envelopes() const noexcept {
        return has_disjoint_row_envelopes_;
    }

private:
    RowwiseViewAnalysis(RowwiseAddressFootprint footprint,
                        int64_t row_stride,
                        int64_t column_stride,
                        int64_t max_element_offset,
                        InjectivityClassification injectivity,
                        bool has_disjoint_row_envelopes) noexcept;

    friend class AliasUtilsFactory;

    RowwiseAddressFootprint footprint_;
    int64_t row_stride_{};
    int64_t column_stride_{};
    int64_t max_element_offset_{};
    InjectivityClassification injectivity_{InjectivityClassification::kUnknown};
    bool has_disjoint_row_envelopes_{};
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
AM_NODISCARD inline bool ByteRangesOverlap(const ByteAddressRange& lhs,
                                           const ByteAddressRange& rhs) noexcept {
    if (lhs.begin >= lhs.end || rhs.begin >= rhs.end) {
        return false;
    }
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

/// @brief Builds a range from caller-proven contiguous storage geometry.
///
/// The range is `[data, data + element_count * element_size)`. An empty view
/// (`element_count == 0`) produces an empty range at `data`; a non-empty view
/// requires non-null data and a non-zero element size. This raw overload has
/// no TensorView metadata and cannot verify that a tensor is contiguous; the
/// caller must establish that its supplied count describes consecutive elements.
/// Use the TensorView overloads when a view is available.
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

/// @brief Builds the range of a valid row-major contiguous tensor view.
///
/// Performs release-build view validation, checked strided-footprint geometry,
/// and row-major contiguity validation before returning the footprint envelope.
/// Rank-0 is one element; zero-element views return an empty range and may have
/// null data.
StatusOr<ByteAddressRange> BuildContiguousByteRange(const TensorView& tensor,
                                                    std::string_view context) noexcept;

/// @see BuildContiguousByteRange(const TensorView&, std::string_view)
StatusOr<ByteAddressRange> BuildContiguousByteRange(const MutableTensorView& tensor,
                                                    std::string_view context) noexcept;

/// @brief Requires a mutable output and read input to be provably disjoint.
///
/// Contiguous counterpart of ValidateRowwiseDisjoint with the same message
/// convention: proven overlap is InvalidArgument. Contiguous ranges are exact,
/// so there is no undecidable case to report as Unimplemented.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param output Address range of the mutable output being validated.
/// @param output_role Output role (e.g. "output") used in the message.
/// @param input Address range of the read-only input compared against.
/// @param input_role Input role (e.g. "weight") used in the message.
/// @return Ok when the ranges are provably disjoint.
Status ValidateContiguousDisjoint(std::string_view kernel_name,
                                  const ByteAddressRange& output,
                                  std::string_view output_role,
                                  const ByteAddressRange& input,
                                  std::string_view input_role) noexcept;

/// @brief Analyzes a valid rank >= 1 tensor as a row-wise `[rows, columns]` view.
///
/// The last dimension is the column dimension and leading dimensions are
/// flattened into rows. A rank-1 view is therefore one row with a derived
/// `row_stride` of zero; that derived stride is not an actual tensor stride and
/// is legal. View validity is checked first, matching the strided-footprint
/// overloads; for non-empty views this then validates positive actual strides,
/// collapsible leading axes, signed element-offset representability, and the
/// byte-address footprint in that order. Empty views retain the existing no-op
/// contract and do not require data or positive strides, but must still be
/// valid views.
///
/// This function deliberately does not reject self-overlapping row envelopes:
/// immutable inputs may legally reuse elements. Mutable-output callers must
/// apply ValidateRowwiseOutputLayout to the returned facts.
///
/// @param tensor Valid view whose last axis becomes the column dimension.
/// @param context Caller name used as the error-message prefix.
/// @return The checked row-wise facts, or InvalidArgument / Unimplemented when
///         the view or its geometry is not supported.
StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseView(const TensorView& tensor,
                                                 std::string_view context) noexcept;

/// @see AnalyzeRowwiseView(const TensorView&, std::string_view)
StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseView(const MutableTensorView& tensor,
                                                 std::string_view context) noexcept;

/// @brief Analyzes a valid rank-1 tensor as a `[length, 1]` row-wise column vector.
///
/// This explicit adapter preserves per-element stride gaps when a vector is
/// compared against row-wise outputs, such as RoPE `position_ids`. View
/// validity and positive strides are checked as in AnalyzeRowwiseView, and an
/// empty vector keeps the same no-op contract.
///
/// @param tensor Valid rank-1 view whose elements become rows of one column.
/// @param context Caller name used as the error-message prefix.
/// @return The checked row-wise facts, or InvalidArgument when the view is
///         invalid, is not rank-1, or has unsupported geometry.
StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseColumnVector(
        const TensorView& tensor,
        std::string_view context) noexcept;

/// @see AnalyzeRowwiseColumnVector(const TensorView&, std::string_view)
StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseColumnVector(
        const MutableTensorView& tensor,
        std::string_view context) noexcept;

/// @brief Applies the supported mutable-output row-envelope policy.
///
/// Row-envelope disjointness is intentionally stricter than a proof of actual
/// element collisions: it keeps output writes in separate address regions even
/// when column-stride holes would otherwise make overlap undecidable.
Status ValidateRowwiseOutputLayout(std::string_view context,
                                   const RowwiseViewAnalysis& analysis) noexcept;

/// @brief Requires a mutable output and read input to be provably disjoint.
///
/// Proven overlap is InvalidArgument, while overlap that column-stride holes
/// prevent this helper from deciding is reported as Unimplemented instead of
/// claiming a violation. Callers decide which pairs to validate and grant
/// exemptions such as exact in-place before calling.
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

/// @brief Address geometry and proven layout properties of a strided view.
///
/// `envelope` spans from the base address past the last logical element and can
/// contain holes that are not logical elements of the view. `is_dense` reports
/// that the logical elements exactly fill the envelope without holes or
/// repeats; that is weaker than row-major contiguity, since a transposed layout
/// can be dense without being contiguous. Instances are produced only by the
/// checked builders in this component and expose read-only facts to callers.
class StridedAddressFootprint {
public:
    AM_NODISCARD ByteAddressRange envelope() const noexcept {
        return envelope_;
    }

    AM_NODISCARD int64_t logical_element_count() const noexcept {
        return logical_element_count_;
    }

    AM_NODISCARD int64_t max_element_offset() const noexcept {
        return max_element_offset_;
    }

    AM_NODISCARD InjectivityClassification injectivity() const noexcept {
        return injectivity_;
    }

    AM_NODISCARD bool is_dense() const noexcept {
        return is_dense_;
    }

    AM_NODISCARD bool is_empty() const noexcept {
        return is_empty_;
    }

private:
    StridedAddressFootprint(ByteAddressRange envelope,
                            int64_t logical_element_count,
                            int64_t max_element_offset,
                            InjectivityClassification injectivity,
                            bool is_dense,
                            bool is_empty) noexcept;

    friend class AliasUtilsFactory;

    ByteAddressRange envelope_{};
    int64_t logical_element_count_{};
    int64_t max_element_offset_{};
    InjectivityClassification injectivity_{InjectivityClassification::kUnknown};
    bool is_dense_{};
    bool is_empty_{};
};

/// @brief Builds checked address geometry from caller-supplied strided data.
///
/// This raw overload verifies only the supplied pointer/shape/stride geometry;
/// it does not validate TensorView metadata, dtype, alignment, or lifetime.
/// Callers with TensorViews should use the view overloads. A view with any zero
/// extent produces an empty footprint whose envelope is
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

/// @brief Builds checked address geometry from a valid immutable tensor view.
///
/// Binds data, shape, strides, and item size coherently after validating the
/// view in release builds. Rank-0 and empty-view behavior match the raw helper.
StatusOr<StridedAddressFootprint> BuildStridedAddressFootprint(
        const TensorView& tensor,
        std::string_view context) noexcept;

/// @see BuildStridedAddressFootprint(const TensorView&, std::string_view)
StatusOr<StridedAddressFootprint> BuildStridedAddressFootprint(
        const MutableTensorView& tensor,
        std::string_view context) noexcept;

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
AM_NODISCARD bool HaveIdenticalViewMapping(const TensorView& input,
                                           const MutableTensorView& output) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_ALIAS_UTILS_H
