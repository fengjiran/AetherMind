#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/base/macros.h"
#include "aethermind/base/shape_and_stride.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind::cpu::detail {

RowwiseAddressFootprint::RowwiseAddressFootprint(ByteAddressRange envelope,
                                                 std::uintptr_t row_stride_bytes,
                                                 std::uintptr_t row_envelope_bytes,
                                                 std::uintptr_t column_stride_bytes,
                                                 std::uintptr_t element_size_bytes,
                                                 int64_t row_count,
                                                 int64_t column_count,
                                                 bool is_empty) noexcept
    : envelope_(envelope),
      row_stride_bytes_(row_stride_bytes),
      row_envelope_bytes_(row_envelope_bytes),
      column_stride_bytes_(column_stride_bytes),
      element_size_bytes_(element_size_bytes),
      row_count_(row_count),
      column_count_(column_count),
      is_empty_(is_empty) {
    // Emptiness is definitional: a zero row or column count leaves the byte
    // geometry fields meaningless. Keep the explicit flag in lockstep with the
    // counts so a future caller cannot fabricate an inconsistent footprint.
    AM_DCHECK(is_empty == (row_count == 0 || column_count == 0));
}

RowwiseViewAnalysis::RowwiseViewAnalysis(RowwiseAddressFootprint footprint,
                                         int64_t row_stride,
                                         int64_t column_stride,
                                         int64_t max_element_offset,
                                         InjectivityClassification injectivity,
                                         bool has_disjoint_row_envelopes) noexcept
    : footprint_(footprint),
      row_stride_(row_stride),
      column_stride_(column_stride),
      max_element_offset_(max_element_offset),
      injectivity_(injectivity),
      has_disjoint_row_envelopes_(has_disjoint_row_envelopes) {}

StridedAddressFootprint::StridedAddressFootprint(ByteAddressRange envelope,
                                                 int64_t logical_element_count,
                                                 int64_t max_element_offset,
                                                 InjectivityClassification injectivity,
                                                 bool is_dense,
                                                 bool is_empty) noexcept
    : envelope_(envelope),
      logical_element_count_(logical_element_count),
      max_element_offset_(max_element_offset),
      injectivity_(injectivity),
      is_dense_(is_dense),
      is_empty_(is_empty) {}

class AliasUtilsFactory {
public:
    static RowwiseAddressFootprint MakeRowwiseAddressFootprint(
            ByteAddressRange envelope,
            std::uintptr_t row_stride_bytes,
            std::uintptr_t row_envelope_bytes,
            std::uintptr_t column_stride_bytes,
            std::uintptr_t element_size_bytes,
            int64_t row_count,
            int64_t column_count,
            bool is_empty) noexcept {
        return {envelope, row_stride_bytes, row_envelope_bytes,
                column_stride_bytes, element_size_bytes, row_count,
                column_count, is_empty};
    }

    static RowwiseViewAnalysis MakeRowwiseViewAnalysis(
            RowwiseAddressFootprint footprint,
            int64_t row_stride,
            int64_t column_stride,
            int64_t max_element_offset,
            InjectivityClassification injectivity,
            bool has_disjoint_row_envelopes) noexcept {
        return {footprint, row_stride,
                column_stride, max_element_offset, injectivity,
                has_disjoint_row_envelopes};
    }

    static StridedAddressFootprint MakeStridedAddressFootprint(
            ByteAddressRange envelope,
            int64_t logical_element_count,
            int64_t max_element_offset,
            InjectivityClassification injectivity,
            bool is_dense,
            bool is_empty) noexcept {
        return {envelope, logical_element_count, max_element_offset,
                injectivity, is_dense, is_empty};
    }
};

namespace {

enum class OverlapClassification : uint8_t {
    kProvenDisjoint,
    kProvenOverlap,
    kUnknown
};

AM_NODISCARD InjectivityClassification ClassifyLayoutInjectivity(std::span<const int64_t> shape,
                                                                 std::span<const int64_t> strides) noexcept {
    AM_DCHECK(shape.size() == strides.size());
    AM_DCHECK(shape.size() <= ShapeAndStride::kMaxRank);

    // Release-build fail-safe: malformed geometry must not index past the
    // fixed-rank axis buffer, so a violated precondition is reported as
    // undecidable instead of corrupting memory.
    if (shape.size() != strides.size() || shape.size() > ShapeAndStride::kMaxRank) {
        return InjectivityClassification::kUnknown;
    }

    struct Axis {
        int64_t extent;
        int64_t stride;
    };

    std::array<Axis, ShapeAndStride::kMaxRank> axes{};
    size_t axis_count = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 1) {
            axes[axis_count++] = Axis{.extent = shape[i], .stride = strides[i]};
        }
    }

    const auto ordered = std::span{axes}.first(axis_count);
    std::ranges::sort(ordered,
                      [](const Axis& lhs, const Axis& rhs) {
                          return lhs.stride < rhs.stride;
                      });

    int64_t covered_span = 1;
    bool lower_axes_are_dense = true;
    for (const auto& [extent, stride]: ordered) {
        if (stride < covered_span) {
            // Dense lower axes reach every offset in [0, covered_span), so this
            // stride is hit twice: once by the lower axes and once by index 1 of
            // the current axis. With holes the collision is not provable.
            return lower_axes_are_dense ? InjectivityClassification::kProvenNonInjective
                                        : InjectivityClassification::kUnknown;
        }

        if (stride != covered_span) {
            lower_axes_are_dense = false;
        }

        int64_t axis_span = 0;
        int64_t next_span = 0;
        if (CheckOverflowMul(extent - 1, stride, &axis_span) ||
            CheckOverflowAdd(covered_span, axis_span, &next_span)) {
            return InjectivityClassification::kUnknown;
        }
        covered_span = next_span;
    }
    return InjectivityClassification::kProvenInjective;
}

AM_NODISCARD bool TryConvertToUintptr(int64_t value, std::uintptr_t* result) noexcept {
    if (value < 0 ||
        static_cast<std::uintmax_t>(value) > std::numeric_limits<std::uintptr_t>::max()) {
        return false;
    }

    *result = static_cast<std::uintptr_t>(value);
    return true;
}

Status InvalidRowwiseGeometry(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid row-wise address geometry");
}

Status InvalidRowwiseView(std::string_view context, std::string_view message) {
    return Status::InvalidArgument(std::string(context) + " " + std::string(message));
}

Status InvalidContiguousGeometry(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid contiguous address geometry");
}

Status InvalidStridedGeometry(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid strided address geometry");
}

Status InvalidTensorView(std::string_view context) {
    return Status::InvalidArgument(std::string(context) + " requires a valid tensor view");
}

Status NonContiguousTensorView(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " requires a row-major contiguous tensor view");
}

Status AddressComputationOverflow(std::string_view context) {
    return Status::InvalidArgument(std::string(context) + " address range overflow");
}

AM_NODISCARD ByteAddressRange RowAddressEnvelope(const RowwiseAddressFootprint& footprint,
                                                 int64_t row) noexcept {
    // BuildRowwiseAddressFootprintImpl proved that the final row end is representable.
    const std::uintptr_t row_offset = static_cast<std::uintptr_t>(row) * footprint.row_stride_bytes();
    const std::uintptr_t begin = footprint.envelope().begin + row_offset;
    return ByteAddressRange{.begin = begin, .end = begin + footprint.row_envelope_bytes()};
}

AM_NODISCARD bool RowsHaveColumnStrideHoles(const RowwiseAddressFootprint& footprint) noexcept {
    return footprint.column_count() > 1 &&
           footprint.column_stride_bytes() != footprint.element_size_bytes();
}

StatusOr<RowwiseAddressFootprint> BuildRowwiseAddressFootprintImpl(
        const void* data,
        int64_t row_count,
        int64_t column_count,
        int64_t row_stride,
        int64_t column_stride,
        size_t element_size,
        std::string_view context) noexcept {
    if (row_count < 0 || column_count < 0 || row_stride < 0 || column_stride < 0 ||
        element_size == 0) {
        return InvalidRowwiseGeometry(context);
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (row_count == 0 || column_count == 0) {
        return AliasUtilsFactory::MakeRowwiseAddressFootprint(
                ByteAddressRange{.begin = begin, .end = begin},
                0, 0, 0, 0, row_count,
                column_count, true);
    }

    if (data == nullptr) {
        return InvalidRowwiseGeometry(context);
    }

    std::uintptr_t row_count_minus_one = 0;
    std::uintptr_t column_count_minus_one = 0;
    std::uintptr_t row_stride_elements = 0;
    std::uintptr_t column_stride_elements = 0;
    if (!TryConvertToUintptr(row_count - 1, &row_count_minus_one) ||
        !TryConvertToUintptr(column_count - 1, &column_count_minus_one) ||
        !TryConvertToUintptr(row_stride, &row_stride_elements) ||
        !TryConvertToUintptr(column_stride, &column_stride_elements) ||
        element_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressComputationOverflow(context);
    }

    const std::uintptr_t element_size_bytes = element_size;
    std::uintptr_t last_col_offset = 0;
    std::uintptr_t row_envelope_bytes = 0;
    std::uintptr_t row_stride_bytes = 0;
    std::uintptr_t column_stride_bytes = 0;
    std::uintptr_t last_row_offset = 0;
    if (CheckOverflowMul(column_count_minus_one, column_stride_elements, &last_col_offset) ||
        CheckOverflowMul(last_col_offset, element_size_bytes, &row_envelope_bytes) ||
        CheckOverflowAdd(row_envelope_bytes, element_size_bytes, &row_envelope_bytes) ||
        CheckOverflowMul(row_stride_elements, element_size_bytes, &row_stride_bytes) ||
        CheckOverflowMul(column_stride_elements, element_size_bytes, &column_stride_bytes) ||
        CheckOverflowMul(row_count_minus_one, row_stride_bytes, &last_row_offset)) {
        return AddressComputationOverflow(context);
    }

    std::uintptr_t last_row_begin = 0;
    std::uintptr_t end = 0;
    if (CheckOverflowAdd(begin, last_row_offset, &last_row_begin) ||
        CheckOverflowAdd(last_row_begin, row_envelope_bytes, &end)) {
        return AddressComputationOverflow(context);
    }

    return AliasUtilsFactory::MakeRowwiseAddressFootprint(
            ByteAddressRange{.begin = begin, .end = end}, row_stride_bytes, row_envelope_bytes,
            column_stride_bytes, element_size_bytes, row_count, column_count, false);
}

StatusOr<RowwiseViewAnalysis> BuildRowwiseViewAnalysis(const void* data,
                                                       int64_t row_count,
                                                       int64_t column_count,
                                                       int64_t row_stride,
                                                       int64_t column_stride,
                                                       size_t element_size,
                                                       std::string_view context) noexcept {
    if (row_count < 0 || column_count < 0 || row_stride < 0 || column_stride < 0) {
        return InvalidRowwiseGeometry(context);
    }

    int64_t max_element_offset = 0;
    int64_t last_column_offset = 0;
    if (row_count > 0 && column_count > 0) {
        int64_t row_offset = 0;
        if (CheckOverflowMul(row_count - 1, row_stride, &row_offset) ||
            CheckOverflowMul(column_count - 1, column_stride, &last_column_offset) ||
            CheckOverflowAdd(row_offset, last_column_offset, &max_element_offset)) {
            return InvalidRowwiseView(context, "element offset overflow");
        }
    }

    AM_ASSIGN_OR_RETURN(const RowwiseAddressFootprint footprint,
                        BuildRowwiseAddressFootprintImpl(data, row_count, column_count,
                                                         row_stride, column_stride, element_size, context));

    const bool is_empty = row_count == 0 || column_count == 0;
    const std::array<int64_t, 2> shape = {row_count, column_count};
    const std::array<int64_t, 2> strides = {row_stride, column_stride};
    const bool has_disjoint_row_envelopes = is_empty || row_count <= 1 || row_stride > last_column_offset;
    return AliasUtilsFactory::MakeRowwiseViewAnalysis(
            footprint, row_stride, column_stride,
            max_element_offset,
            is_empty ? InjectivityClassification::kProvenInjective
                     : ClassifyLayoutInjectivity(shape, strides),
            has_disjoint_row_envelopes);
}

template<typename TensorLike>
StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseViewImpl(const TensorLike& tensor,
                                                     std::string_view context) noexcept {
    if (!tensor.is_valid()) {
        return InvalidTensorView(context);
    }

    if (tensor.rank() < 1) {
        return InvalidRowwiseView(context, "requires rank >= 1");
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_count, ComputeFlattenedRowCount(tensor, context));
    const int64_t column_count = tensor.dim(tensor.rank() - 1);
    const int64_t row_stride = tensor.rank() == 1 ? 0 : tensor.stride(tensor.rank() - 2);
    const int64_t column_stride = tensor.stride(tensor.rank() - 1);
    if (row_count == 0 || column_count == 0) {
        return BuildRowwiseViewAnalysis(tensor.data(), row_count, column_count, row_stride,
                                        column_stride, tensor.itemsize(), context);
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            tensor, std::string(context) + " requires positive strides"));
    if (tensor.rank() > 2) {
        AM_RETURN_IF_ERROR(ValidateFlattenableLeadingAxes(tensor, context));
    }
    return BuildRowwiseViewAnalysis(tensor.data(), row_count, column_count, row_stride,
                                    column_stride, tensor.itemsize(), context);
}

template<typename TensorLike>
StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseColumnVectorImpl(const TensorLike& tensor,
                                                             std::string_view context) noexcept {
    if (!tensor.is_valid()) {
        return InvalidTensorView(context);
    }

    if (tensor.rank() != 1) {
        return InvalidRowwiseView(context, "requires rank-1 tensor");
    }

    const int64_t row_count = tensor.dim(0);
    const int64_t row_stride = tensor.stride(0);
    if (row_count == 0) {
        return BuildRowwiseViewAnalysis(tensor.data(), 0, 1, row_stride, 1,
                                        tensor.itemsize(), context);
    }

    AM_RETURN_IF_ERROR(ValidatePositiveStrides(
            tensor, std::string(context) + " requires positive strides"));
    return BuildRowwiseViewAnalysis(tensor.data(), row_count, 1, row_stride, 1,
                                    tensor.itemsize(), context);
}

AM_NODISCARD OverlapClassification ClassifyRowwiseOverlap(const RowwiseAddressFootprint& lhs,
                                                          const RowwiseAddressFootprint& rhs) noexcept {
    if (lhs.is_empty() || rhs.is_empty() ||
        !ByteRangesOverlap(lhs.envelope(), rhs.envelope())) {
        return OverlapClassification::kProvenDisjoint;
    }

    int64_t lhs_row = 0;
    int64_t rhs_row = 0;
    bool row_envelopes_overlap = false;
    while (lhs_row < lhs.row_count() && rhs_row < rhs.row_count()) {
        const ByteAddressRange lhs_range = RowAddressEnvelope(lhs, lhs_row);
        const ByteAddressRange rhs_range = RowAddressEnvelope(rhs, rhs_row);
        if (ByteRangesOverlap(lhs_range, rhs_range)) {
            row_envelopes_overlap = true;
            break;
        }

        if (lhs_range.end <= rhs_range.begin) {
            ++lhs_row;
        } else {
            ++rhs_row;
        }
    }

    if (!row_envelopes_overlap) {
        return OverlapClassification::kProvenDisjoint;
    }

    if (lhs.envelope().begin == rhs.envelope().begin ||
        (!RowsHaveColumnStrideHoles(lhs) && !RowsHaveColumnStrideHoles(rhs))) {
        return OverlapClassification::kProvenOverlap;
    }
    return OverlapClassification::kUnknown;
}

AM_NODISCARD OverlapClassification ClassifyStridedOverlap(const StridedAddressFootprint& lhs,
                                                          const StridedAddressFootprint& rhs) noexcept {
    if (lhs.is_empty() || rhs.is_empty() ||
        !ByteRangesOverlap(lhs.envelope(), rhs.envelope())) {
        return OverlapClassification::kProvenDisjoint;
    }

    if ((lhs.is_dense() && rhs.is_dense()) || lhs.envelope().begin == rhs.envelope().begin) {
        return OverlapClassification::kProvenOverlap;
    }
    return OverlapClassification::kUnknown;
}

} // namespace

StatusOr<ByteAddressRange> BuildContiguousByteRange(const void* data,
                                                    int64_t element_count,
                                                    size_t element_size,
                                                    std::string_view context) noexcept {
    if (element_count < 0 || element_size == 0) {
        return InvalidContiguousGeometry(context);
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (element_count == 0) {
        return ByteAddressRange{.begin = begin, .end = begin};
    }

    if (data == nullptr) {
        return InvalidContiguousGeometry(context);
    }

    std::uintptr_t element_count_value = 0;
    if (!TryConvertToUintptr(element_count, &element_count_value) ||
        element_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressComputationOverflow(context);
    }

    std::uintptr_t span_bytes = 0;
    std::uintptr_t end = 0;
    if (CheckOverflowMul(element_count_value, element_size, &span_bytes) ||
        CheckOverflowAdd(begin, span_bytes, &end)) {
        return AddressComputationOverflow(context);
    }

    return ByteAddressRange{.begin = begin, .end = end};
}

StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseView(const TensorView& tensor,
                                                 std::string_view context) noexcept {
    return AnalyzeRowwiseViewImpl(tensor, context);
}

StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseView(const MutableTensorView& tensor,
                                                 std::string_view context) noexcept {
    return AnalyzeRowwiseViewImpl(tensor, context);
}

StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseColumnVector(const TensorView& tensor,
                                                         std::string_view context) noexcept {
    return AnalyzeRowwiseColumnVectorImpl(tensor, context);
}

StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseColumnVector(const MutableTensorView& tensor,
                                                         std::string_view context) noexcept {
    return AnalyzeRowwiseColumnVectorImpl(tensor, context);
}

Status ValidateRowwiseOutputLayout(std::string_view context,
                                   const RowwiseViewAnalysis& analysis) noexcept {
    if (!analysis.has_disjoint_row_envelopes()) {
        return Status::InvalidArgument(
                std::string(context) + " requires disjoint output row envelopes");
    }
    return ValidateLayoutInjectivity(context, analysis.injectivity(), "output");
}

Status ValidateRowwiseDisjoint(std::string_view kernel_name,
                               const RowwiseAddressFootprint& output,
                               std::string_view output_role,
                               const RowwiseAddressFootprint& input,
                               std::string_view input_role) noexcept {
    switch (ClassifyRowwiseOverlap(output, input)) {
        case OverlapClassification::kProvenDisjoint:
            return Status::Ok();
        case OverlapClassification::kProvenOverlap:
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " + std::string(output_role) +
                    " must not overlap " + std::string(input_role));
        case OverlapClassification::kUnknown:
            return Status::Unimplemented(
                    std::string(kernel_name) + " cannot prove " +
                    std::string(output_role) + " is disjoint from " + std::string(input_role) +
                    " for the requested strided layouts");
    }
    return Status::Internal(std::string(kernel_name) +
                            " row-wise overlap classification is invalid");
}

Status ValidateRowwiseDisjointFromContiguous(std::string_view kernel_name,
                                             const RowwiseAddressFootprint& output,
                                             std::string_view output_role,
                                             const ByteAddressRange& input,
                                             std::string_view input_role) noexcept {
    if (input.begin > input.end ||
        input.end - input.begin >
                static_cast<std::uintptr_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow(std::string(kernel_name) +
                                " contiguous input range is not representable");
    }

    const auto input_bytes = static_cast<int64_t>(input.end - input.begin);
    AM_ASSIGN_OR_RETURN(
            const RowwiseAddressFootprint input_footprint,
            BuildRowwiseAddressFootprintImpl(
                    reinterpret_cast<const void*>(input.begin), 1, input_bytes,
                    input_bytes, 1, 1, input_role));
    return ValidateRowwiseDisjoint(kernel_name, output, output_role,
                                   input_footprint, input_role);
}

StatusOr<StridedAddressFootprint> BuildStridedAddressFootprint(
        const void* data,
        std::span<const int64_t> shape,
        std::span<const int64_t> strides,
        size_t element_size,
        std::string_view context) noexcept {
    if (shape.size() != strides.size() || shape.size() > ShapeAndStride::kMaxRank ||
        element_size == 0) {
        return InvalidStridedGeometry(context);
    }

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0 || strides[i] < 0) {
            return InvalidStridedGeometry(context);
        }
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    int64_t numel = 1;
    int64_t max_offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        const int64_t extent = shape[i];
        if (extent == 0) {
            numel = 0;
            max_offset = 0;
            break;
        }

        int64_t next_numel = 0;
        int64_t axis_offset = 0;
        int64_t next_offset = 0;
        if (CheckOverflowMul(numel, extent, &next_numel) ||
            CheckOverflowMul(extent - 1, strides[i], &axis_offset) ||
            CheckOverflowAdd(max_offset, axis_offset, &next_offset)) {
            return AddressComputationOverflow(context);
        }
        numel = next_numel;
        max_offset = next_offset;
    }

    if (numel == 0) {
        return AliasUtilsFactory::MakeStridedAddressFootprint(
                ByteAddressRange{.begin = begin, .end = begin}, 0, 0,
                InjectivityClassification::kProvenInjective, false, true);
    }

    if (data == nullptr) {
        return InvalidStridedGeometry(context);
    }

    const auto injectivity = ClassifyLayoutInjectivity(shape, strides);
    int64_t offset_plus_one = 0;
    if (CheckOverflowAdd(max_offset, int64_t{1}, &offset_plus_one)) {
        return AddressComputationOverflow(context);
    }

    std::uintptr_t max_offset_value = 0;
    if (!TryConvertToUintptr(max_offset, &max_offset_value) ||
        element_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressComputationOverflow(context);
    }

    std::uintptr_t end = 0;
    if (CheckOverflowMul(max_offset_value, element_size, &end) ||
        CheckOverflowAdd(begin, end, &end) ||
        CheckOverflowAdd(end, element_size, &end)) {
        return AddressComputationOverflow(context);
    }

    // Injective elements confined to [0, max_offset] fill the envelope exactly
    // when their count equals its length, leaving no holes.
    const bool is_dense = injectivity == InjectivityClassification::kProvenInjective &&
                          offset_plus_one == numel;
    return AliasUtilsFactory::MakeStridedAddressFootprint(
            ByteAddressRange{.begin = begin, .end = end}, numel, max_offset, injectivity,
            is_dense, false);
}

// Internal view adapters. The anonymous namespace keeps them and their template
// instantiations out of the exported symbol table.
namespace {

template<typename TensorLike>
StatusOr<StridedAddressFootprint> BuildStridedAddressFootprintImpl(
        const TensorLike& tensor,
        std::string_view context) noexcept {
    if (!tensor.is_valid()) {
        return InvalidTensorView(context);
    }
    return BuildStridedAddressFootprint(
            tensor.data(), tensor.shape(), tensor.strides(),
            tensor.itemsize(), context);
}

// Walks row-major contiguity explicitly instead of calling
// TensorView::is_contiguous() on purpose. The base-layer walk aborts the
// process through AM_CHECK when the expected-stride product overflows int64,
// which untrusted view metadata can trigger; binding-time validation must
// report InvalidArgument instead of terminating. The two walks are otherwise
// identical (extent-1 axes are skipped, rank-0 is contiguous), so keep them in
// sync if ShapeAndStride::is_contiguous() ever changes.
template<typename TensorLike>
StatusOr<bool> IsRowMajorContiguousChecked(const TensorLike& tensor,
                                           std::string_view context) noexcept {
    int64_t expected_stride = 1;
    for (int32_t axis = tensor.rank() - 1; axis >= 0; --axis) {
        const int64_t extent = tensor.dim(axis);
        if (extent == 1) {
            continue;
        }

        if (tensor.stride(axis) != expected_stride) {
            return false;
        }

        int64_t next_expected_stride = 0;
        if (CheckOverflowMul(expected_stride, extent, &next_expected_stride)) {
            return Status::InvalidArgument(
                    std::string(context) + " contiguous stride product overflow");
        }
        expected_stride = next_expected_stride;
    }
    return true;
}

template<typename TensorLike>
StatusOr<ByteAddressRange> BuildContiguousByteRangeImpl(const TensorLike& tensor,
                                                        std::string_view context) noexcept {
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint footprint,
                        BuildStridedAddressFootprintImpl(tensor, context));
    AM_ASSIGN_OR_RETURN(const bool is_contiguous,
                        IsRowMajorContiguousChecked(tensor, context));
    if (!is_contiguous) {
        return NonContiguousTensorView(context);
    }
    return footprint.envelope();
}

} // namespace

StatusOr<StridedAddressFootprint>
BuildStridedAddressFootprint(const TensorView& tensor,
                             std::string_view context) noexcept {
    return BuildStridedAddressFootprintImpl(tensor, context);
}

StatusOr<StridedAddressFootprint>
BuildStridedAddressFootprint(const MutableTensorView& tensor,
                             std::string_view context) noexcept {
    return BuildStridedAddressFootprintImpl(tensor, context);
}

StatusOr<ByteAddressRange> BuildContiguousByteRange(const TensorView& tensor,
                                                    std::string_view context) noexcept {
    return BuildContiguousByteRangeImpl(tensor, context);
}

StatusOr<ByteAddressRange> BuildContiguousByteRange(const MutableTensorView& tensor,
                                                    std::string_view context) noexcept {
    return BuildContiguousByteRangeImpl(tensor, context);
}

Status ValidateContiguousDisjoint(std::string_view kernel_name,
                                  const ByteAddressRange& output,
                                  std::string_view output_role,
                                  const ByteAddressRange& input,
                                  std::string_view input_role) noexcept {
    if (!ByteRangesOverlap(output, input)) {
        return Status::Ok();
    }
    return Status::InvalidArgument(
            std::string(kernel_name) + " " + std::string(output_role) +
            " must not overlap " + std::string(input_role));
}

Status ValidateLayoutInjectivity(std::string_view kernel_name,
                                 InjectivityClassification injectivity,
                                 std::string_view role) noexcept {
    switch (injectivity) {
        case InjectivityClassification::kProvenInjective:
            return Status::Ok();
        case InjectivityClassification::kProvenNonInjective:
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " + std::string(role) +
                    " logical elements must not overlap");
        case InjectivityClassification::kUnknown:
            return Status::Unimplemented(
                    std::string(kernel_name) + " cannot prove " + std::string(role) +
                    " maps distinct coordinates to distinct offsets");
    }
    return Status::Internal(std::string(kernel_name) +
                            " layout injectivity classification is invalid");
}

Status ValidateStridedDisjoint(std::string_view kernel_name,
                               const StridedAddressFootprint& output,
                               std::string_view output_role,
                               const StridedAddressFootprint& input,
                               std::string_view input_role) noexcept {
    switch (ClassifyStridedOverlap(output, input)) {
        case OverlapClassification::kProvenDisjoint:
            return Status::Ok();
        case OverlapClassification::kProvenOverlap:
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " + std::string(output_role) +
                    " must not overlap " + std::string(input_role));
        case OverlapClassification::kUnknown:
            return Status::Unimplemented(
                    std::string(kernel_name) + " cannot prove " + std::string(output_role) +
                    " is disjoint from " + std::string(input_role) +
                    " for the requested strided layouts");
    }
    return Status::Internal(std::string(kernel_name) +
                            " address footprint overlap classification is invalid");
}

AM_NODISCARD bool HaveIdenticalViewMapping(const TensorView& input,
                                           const MutableTensorView& output) noexcept {
    if (input.data() != output.data() || input.dtype() != output.dtype() ||
        input.rank() != output.rank()) {
        return false;
    }

    for (int32_t i = 0; i < input.rank(); ++i) {
        if (input.dim(i) != output.dim(i) || input.stride(i) != output.stride(i)) {
            return false;
        }
    }
    return true;
}

} // namespace aethermind::cpu::detail
