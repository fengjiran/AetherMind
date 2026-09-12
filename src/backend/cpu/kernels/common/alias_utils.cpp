#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
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

namespace aethermind::cpu::detail {
namespace {

bool TryConvertToUintptr(int64_t value, std::uintptr_t* result) noexcept {
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

Status InvalidContiguousGeometry(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid contiguous address geometry");
}

Status InvalidStridedGeometry(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid strided address geometry");
}

Status AddressComputationOverflow(std::string_view context) {
    return Status::InvalidArgument(std::string(context) + " address range overflow");
}

ByteAddressRange RowAddressEnvelope(const RowwiseAddressFootprint& footprint,
                                    int64_t row) noexcept {
    // BuildRowwiseAddressFootprint proved that the final row end is representable.
    const std::uintptr_t row_offset = static_cast<std::uintptr_t>(row) * footprint.row_stride_bytes;
    const std::uintptr_t begin = footprint.envelope.begin + row_offset;
    return ByteAddressRange{.begin = begin, .end = begin + footprint.row_envelope_bytes};
}

bool RowsHaveColumnStrideHoles(const RowwiseAddressFootprint& footprint) noexcept {
    return footprint.column_count > 1 &&
           footprint.column_stride_bytes != footprint.element_size_bytes;
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

StatusOr<RowwiseAddressFootprint> BuildRowwiseAddressFootprint(
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
        return RowwiseAddressFootprint{
                .envelope = ByteAddressRange{.begin = begin, .end = begin},
                .row_count = row_count,
                .column_count = column_count,
        };
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

    return RowwiseAddressFootprint{
            .envelope = ByteAddressRange{.begin = begin, .end = end},
            .row_stride_bytes = row_stride_bytes,
            .row_envelope_bytes = row_envelope_bytes,
            .column_stride_bytes = column_stride_bytes,
            .element_size_bytes = element_size_bytes,
            .row_count = row_count,
            .column_count = column_count,
    };
}

OverlapClassification ClassifyRowwiseOverlap(const RowwiseAddressFootprint& lhs,
                                             const RowwiseAddressFootprint& rhs) noexcept {
    if (!ByteRangesOverlap(lhs.envelope, rhs.envelope)) {
        return OverlapClassification::kProvenDisjoint;
    }

    int64_t lhs_row = 0;
    int64_t rhs_row = 0;
    bool row_envelopes_overlap = false;
    while (lhs_row < lhs.row_count && rhs_row < rhs.row_count) {
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

    if (lhs.envelope.begin == rhs.envelope.begin ||
        (!RowsHaveColumnStrideHoles(lhs) && !RowsHaveColumnStrideHoles(rhs))) {
        return OverlapClassification::kProvenOverlap;
    }
    return OverlapClassification::kUnknown;
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

InjectivityClassification ClassifyLayoutInjectivity(
        std::span<const int64_t> shape,
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
            if (lower_axes_are_dense) {
                return InjectivityClassification::kProvenNonInjective;
            }
            return InjectivityClassification::kUnknown;
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
        return StridedAddressFootprint{
                .envelope = ByteAddressRange{.begin = begin, .end = begin},
                .logical_element_count = 0,
                .max_element_offset = 0,
                .injectivity = InjectivityClassification::kProvenInjective,
                .is_dense = false,
                .is_empty = true,
        };
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

    return StridedAddressFootprint{
            .envelope = ByteAddressRange{.begin = begin, .end = end},
            .logical_element_count = numel,
            .max_element_offset = max_offset,
            .injectivity = injectivity,
            // Injective elements confined to [0, max_offset] fill the envelope
            // exactly when their count equals its length, leaving no holes.
            .is_dense = injectivity == InjectivityClassification::kProvenInjective &&
                        offset_plus_one == numel,
            .is_empty = false,
    };
}

OverlapClassification ClassifyStridedOverlap(const StridedAddressFootprint& lhs,
                                             const StridedAddressFootprint& rhs) noexcept {
    if (lhs.is_empty || rhs.is_empty || !ByteRangesOverlap(lhs.envelope, rhs.envelope)) {
        return OverlapClassification::kProvenDisjoint;
    }

    if ((lhs.is_dense && rhs.is_dense) || lhs.envelope.begin == rhs.envelope.begin) {
        return OverlapClassification::kProvenOverlap;
    }
    return OverlapClassification::kUnknown;
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

bool HaveIdenticalViewMapping(const TensorView& input,
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
