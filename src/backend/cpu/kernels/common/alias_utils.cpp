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

bool ToAddressValue(int64_t value, std::uintptr_t* result) noexcept {
    if (value < 0 ||
        static_cast<std::uintmax_t>(value) > std::numeric_limits<std::uintptr_t>::max()) {
        return false;
    }

    *result = static_cast<std::uintptr_t>(value);
    return true;
}

Status AddressGeometryError(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid row-wise address geometry");
}

Status ContiguousGeometryError(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid contiguous address geometry");
}

Status FootprintGeometryError(std::string_view context) {
    return Status::InvalidArgument(std::string(context) +
                                   " has invalid strided address geometry");
}

Status AddressRangeOverflow(std::string_view context) {
    return Status::InvalidArgument(std::string(context) + " address range overflow");
}

AddressRange RowAddressRange(const RowwiseAddressLayout& layout,
                             int64_t row) noexcept {
    // BuildRowwiseAddressLayout proved that the final row end is representable.
    const std::uintptr_t row_offset = static_cast<std::uintptr_t>(row) * layout.row_stride_bytes;
    const std::uintptr_t begin = layout.envelope.begin + row_offset;
    return AddressRange{.begin = begin, .end = begin + layout.row_envelope_bytes};
}

bool HasColumnStrideHoles(const RowwiseAddressLayout& layout) noexcept {
    return layout.col_count > 1 && layout.col_stride_bytes != layout.item_size_bytes;
}

} // namespace

StatusOr<AddressRange> BuildContiguousAddressRange(const void* data,
                                                   int64_t element_count,
                                                   size_t item_size,
                                                   std::string_view context) noexcept {
    if (element_count < 0 || item_size == 0) {
        return ContiguousGeometryError(context);
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (element_count == 0) {
        return AddressRange{.begin = begin, .end = begin};
    }

    if (data == nullptr) {
        return ContiguousGeometryError(context);
    }

    std::uintptr_t element_count_value = 0;
    if (!ToAddressValue(element_count, &element_count_value) ||
        item_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressRangeOverflow(context);
    }

    std::uintptr_t span_bytes = 0;
    std::uintptr_t end = 0;
    if (CheckOverflowMul(element_count_value, item_size, &span_bytes) ||
        CheckOverflowAdd(begin, span_bytes, &end)) {
        return AddressRangeOverflow(context);
    }

    return AddressRange{.begin = begin, .end = end};
}

StatusOr<RowwiseAddressLayout> BuildRowwiseAddressLayout(
        const void* data,
        int64_t row_count,
        int64_t col_count,
        int64_t row_stride,
        int64_t col_stride,
        size_t item_size,
        std::string_view context) noexcept {
    if (row_count < 0 || col_count < 0 || row_stride < 0 || col_stride < 0 ||
        item_size == 0) {
        return AddressGeometryError(context);
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (row_count == 0 || col_count == 0) {
        return RowwiseAddressLayout{
                .envelope = AddressRange{.begin = begin, .end = begin},
                .row_count = row_count,
                .col_count = col_count,
        };
    }

    if (data == nullptr) {
        return AddressGeometryError(context);
    }

    std::uintptr_t row_count_minus_one = 0;
    std::uintptr_t col_count_minus_one = 0;
    std::uintptr_t row_stride_elements = 0;
    std::uintptr_t col_stride_elements = 0;
    if (!ToAddressValue(row_count - 1, &row_count_minus_one) ||
        !ToAddressValue(col_count - 1, &col_count_minus_one) ||
        !ToAddressValue(row_stride, &row_stride_elements) ||
        !ToAddressValue(col_stride, &col_stride_elements) ||
        item_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressRangeOverflow(context);
    }

    const std::uintptr_t item_size_bytes = item_size;
    std::uintptr_t last_col_offset = 0;
    std::uintptr_t row_envelope_bytes = 0;
    std::uintptr_t row_stride_bytes = 0;
    std::uintptr_t col_stride_bytes = 0;
    std::uintptr_t last_row_offset = 0;
    if (CheckOverflowMul(col_count_minus_one, col_stride_elements, &last_col_offset) ||
        CheckOverflowMul(last_col_offset, item_size_bytes, &row_envelope_bytes) ||
        CheckOverflowAdd(row_envelope_bytes, item_size_bytes, &row_envelope_bytes) ||
        CheckOverflowMul(row_stride_elements, item_size_bytes, &row_stride_bytes) ||
        CheckOverflowMul(col_stride_elements, item_size_bytes, &col_stride_bytes) ||
        CheckOverflowMul(row_count_minus_one, row_stride_bytes, &last_row_offset)) {
        return AddressRangeOverflow(context);
    }

    std::uintptr_t last_row_begin = 0;
    std::uintptr_t end = 0;
    if (CheckOverflowAdd(begin, last_row_offset, &last_row_begin) ||
        CheckOverflowAdd(last_row_begin, row_envelope_bytes, &end)) {
        return AddressRangeOverflow(context);
    }

    return RowwiseAddressLayout{
            .envelope = AddressRange{.begin = begin, .end = end},
            .row_stride_bytes = row_stride_bytes,
            .row_envelope_bytes = row_envelope_bytes,
            .col_stride_bytes = col_stride_bytes,
            .item_size_bytes = item_size_bytes,
            .row_count = row_count,
            .col_count = col_count,
    };
}

RowwiseLayoutOverlap ClassifyRowwiseLayoutOverlap(const RowwiseAddressLayout& lhs,
                                                  const RowwiseAddressLayout& rhs) noexcept {
    if (!RangesOverlap(lhs.envelope, rhs.envelope)) {
        return RowwiseLayoutOverlap::kDisjoint;
    }

    int64_t lhs_row = 0;
    int64_t rhs_row = 0;
    bool row_envelopes_overlap = false;
    while (lhs_row < lhs.row_count && rhs_row < rhs.row_count) {
        const AddressRange lhs_range = RowAddressRange(lhs, lhs_row);
        const AddressRange rhs_range = RowAddressRange(rhs, rhs_row);
        if (RangesOverlap(lhs_range, rhs_range)) {
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
        return RowwiseLayoutOverlap::kDisjoint;
    }

    if (lhs.envelope.begin == rhs.envelope.begin ||
        (!HasColumnStrideHoles(lhs) && !HasColumnStrideHoles(rhs))) {
        return RowwiseLayoutOverlap::kProvenOverlap;
    }
    return RowwiseLayoutOverlap::kMayOverlap;
}

bool RowwiseLayoutsMayOverlap(const RowwiseAddressLayout& lhs,
                              const RowwiseAddressLayout& rhs) noexcept {
    return ClassifyRowwiseLayoutOverlap(lhs, rhs) != RowwiseLayoutOverlap::kDisjoint;
}

Status ValidateNoRowwiseOverlap(std::string_view kernel_name,
                                const RowwiseAddressLayout& output,
                                std::string_view output_role,
                                const RowwiseAddressLayout& input,
                                std::string_view input_role) noexcept {
    switch (ClassifyRowwiseLayoutOverlap(output, input)) {
        case RowwiseLayoutOverlap::kDisjoint:
            return Status::Ok();
        case RowwiseLayoutOverlap::kProvenOverlap:
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " + std::string(output_role) +
                    " must not overlap " + std::string(input_role));
        case RowwiseLayoutOverlap::kMayOverlap:
            return Status::Unimplemented(
                    std::string(kernel_name) + " cannot prove " +
                    std::string(output_role) + " is disjoint from " + std::string(input_role) +
                    " for the requested strided layouts");
    }
    return Status::Internal(std::string(kernel_name) +
                            " row-wise overlap classification is invalid");
}

LayoutInjectivity ClassifyLayoutInjectivity(std::span<const int64_t> shape,
                                            std::span<const int64_t> strides) noexcept {
    AM_DCHECK(shape.size() == strides.size());
    AM_DCHECK(shape.size() <= ShapeAndStride::kMaxRank);

    struct Axis {
        int64_t extent;
        int64_t stride;
    };

    std::array<Axis, ShapeAndStride::kMaxRank> axes{};
    size_t axis_count = 0;
    for (size_t dim = 0; dim < shape.size(); ++dim) {
        if (shape[dim] > 1) {
            axes[axis_count++] = Axis{.extent = shape[dim], .stride = strides[dim]};
        }
    }

    const auto ordered = std::span{axes}.first(axis_count);
    std::sort(ordered.begin(), ordered.end(),
              [](const Axis& lhs, const Axis& rhs) { return lhs.stride < rhs.stride; });

    int64_t covered_span = 1;
    bool lower_axes_are_dense = true;
    for (const Axis& axis: ordered) {
        if (axis.stride < covered_span) {
            // Dense lower axes reach every offset in [0, covered_span), so this
            // stride is hit twice: once by the lower axes and once by index 1 of
            // the current axis. With holes the collision is not provable.
            if (lower_axes_are_dense) {
                return LayoutInjectivity::kProvenNonInjective;
            }
            return LayoutInjectivity::kUnknown;
        }

        if (axis.stride != covered_span) {
            lower_axes_are_dense = false;
        }

        int64_t axis_span = 0;
        int64_t next_span = 0;
        if (CheckOverflowMul(axis.extent - 1, axis.stride, &axis_span) ||
            CheckOverflowAdd(covered_span, axis_span, &next_span)) {
            return LayoutInjectivity::kUnknown;
        }
        covered_span = next_span;
    }
    return LayoutInjectivity::kProvenInjective;
}

StatusOr<StridedAddressFootprint> BuildStridedAddressFootprint(
        const void* data,
        std::span<const int64_t> shape,
        std::span<const int64_t> strides,
        size_t item_size,
        std::string_view context) noexcept {
    if (shape.size() != strides.size() || shape.size() > ShapeAndStride::kMaxRank ||
        item_size == 0) {
        return FootprintGeometryError(context);
    }

    for (size_t dim = 0; dim < shape.size(); ++dim) {
        if (shape[dim] < 0 || strides[dim] < 0) {
            return FootprintGeometryError(context);
        }
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);

    int64_t numel = 1;
    int64_t max_offset = 0;
    for (size_t dim = 0; dim < shape.size(); ++dim) {
        const int64_t extent = shape[dim];
        if (extent == 0) {
            numel = 0;
            max_offset = 0;
            break;
        }

        int64_t next_numel = 0;
        int64_t axis_offset = 0;
        int64_t next_offset = 0;
        if (CheckOverflowMul(numel, extent, &next_numel) ||
            CheckOverflowMul(extent - 1, strides[dim], &axis_offset) ||
            CheckOverflowAdd(max_offset, axis_offset, &next_offset)) {
            return AddressRangeOverflow(context);
        }
        numel = next_numel;
        max_offset = next_offset;
    }

    if (numel == 0) {
        return StridedAddressFootprint{
                .envelope = AddressRange{.begin = begin, .end = begin},
                .numel = 0,
                .max_offset = 0,
                .injectivity = LayoutInjectivity::kProvenInjective,
                .is_dense = false,
                .is_empty = true,
        };
    }

    if (data == nullptr) {
        return FootprintGeometryError(context);
    }

    const LayoutInjectivity injectivity = ClassifyLayoutInjectivity(shape, strides);

    int64_t offset_plus_one = 0;
    if (CheckOverflowAdd(max_offset, int64_t{1}, &offset_plus_one)) {
        return AddressRangeOverflow(context);
    }

    std::uintptr_t max_offset_value = 0;
    if (!ToAddressValue(max_offset, &max_offset_value) ||
        item_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressRangeOverflow(context);
    }

    std::uintptr_t end = 0;
    if (CheckOverflowMul(max_offset_value, static_cast<std::uintptr_t>(item_size), &end) ||
        CheckOverflowAdd(begin, end, &end) ||
        CheckOverflowAdd(end, static_cast<std::uintptr_t>(item_size), &end)) {
        return AddressRangeOverflow(context);
    }

    return StridedAddressFootprint{
            .envelope = AddressRange{.begin = begin, .end = end},
            .numel = numel,
            .max_offset = max_offset,
            .injectivity = injectivity,
            // Injective elements confined to [0, max_offset] fill the envelope
            // exactly when their count equals its length, leaving no holes.
            .is_dense = injectivity == LayoutInjectivity::kProvenInjective &&
                        offset_plus_one == numel,
            .is_empty = false,
    };
}

FootprintOverlap ClassifyFootprintOverlap(const StridedAddressFootprint& lhs,
                                          const StridedAddressFootprint& rhs) noexcept {
    if (lhs.is_empty || rhs.is_empty || !RangesOverlap(lhs.envelope, rhs.envelope)) {
        return FootprintOverlap::kDisjoint;
    }

    if ((lhs.is_dense && rhs.is_dense) || lhs.envelope.begin == rhs.envelope.begin) {
        return FootprintOverlap::kProvenOverlap;
    }
    return FootprintOverlap::kMayOverlap;
}

Status ValidateInjectiveLayout(std::string_view kernel_name,
                               LayoutInjectivity injectivity,
                               std::string_view role) noexcept {
    switch (injectivity) {
        case LayoutInjectivity::kProvenInjective:
            return Status::Ok();
        case LayoutInjectivity::kProvenNonInjective:
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " + std::string(role) +
                    " logical elements must not overlap");
        case LayoutInjectivity::kUnknown:
            return Status::Unimplemented(
                    std::string(kernel_name) + " cannot prove " + std::string(role) +
                    " maps distinct coordinates to distinct offsets");
    }
    return Status::Internal(std::string(kernel_name) +
                            " layout injectivity classification is invalid");
}

Status ValidateNoFootprintOverlap(std::string_view kernel_name,
                                  const StridedAddressFootprint& output,
                                  std::string_view output_role,
                                  const StridedAddressFootprint& input,
                                  std::string_view input_role) noexcept {
    switch (ClassifyFootprintOverlap(output, input)) {
        case FootprintOverlap::kDisjoint:
            return Status::Ok();
        case FootprintOverlap::kProvenOverlap:
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " + std::string(output_role) +
                    " must not overlap " + std::string(input_role));
        case FootprintOverlap::kMayOverlap:
            return Status::Unimplemented(
                    std::string(kernel_name) + " cannot prove " + std::string(output_role) +
                    " is disjoint from " + std::string(input_role) +
                    " for the requested strided layouts");
    }
    return Status::Internal(std::string(kernel_name) +
                            " address footprint overlap classification is invalid");
}

bool HasIdenticalMapping(const TensorView& input,
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
