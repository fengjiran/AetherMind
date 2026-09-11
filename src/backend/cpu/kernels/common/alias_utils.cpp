#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <limits>
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
    return layout.column_count > 1 && layout.column_stride_bytes != layout.item_size_bytes;
}

} // namespace

StatusOr<RowwiseAddressLayout> BuildRowwiseAddressLayout(
        const void* data,
        int64_t row_count,
        int64_t column_count,
        int64_t row_stride,
        int64_t column_stride,
        size_t item_size,
        std::string_view context) noexcept {
    if (row_count < 0 || column_count < 0 || row_stride < 0 || column_stride < 0 ||
        item_size == 0) {
        return AddressGeometryError(context);
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (row_count == 0 || column_count == 0) {
        return RowwiseAddressLayout{
                .envelope = AddressRange{.begin = begin, .end = begin},
                .row_count = row_count,
                .column_count = column_count,
        };
    }

    if (data == nullptr) {
        return AddressGeometryError(context);
    }

    std::uintptr_t row_count_minus_one = 0;
    std::uintptr_t column_count_minus_one = 0;
    std::uintptr_t row_stride_elements = 0;
    std::uintptr_t column_stride_elements = 0;
    if (!ToAddressValue(row_count - 1, &row_count_minus_one) ||
        !ToAddressValue(column_count - 1, &column_count_minus_one) ||
        !ToAddressValue(row_stride, &row_stride_elements) ||
        !ToAddressValue(column_stride, &column_stride_elements) ||
        item_size > std::numeric_limits<std::uintptr_t>::max()) {
        return AddressRangeOverflow(context);
    }

    const std::uintptr_t item_size_bytes = item_size;
    std::uintptr_t last_column_offset = 0;
    std::uintptr_t row_envelope_bytes = 0;
    std::uintptr_t row_stride_bytes = 0;
    std::uintptr_t column_stride_bytes = 0;
    std::uintptr_t last_row_offset = 0;
    if (CheckOverflowMul(column_count_minus_one, column_stride_elements, &last_column_offset) ||
        CheckOverflowMul(last_column_offset, item_size_bytes, &row_envelope_bytes) ||
        CheckOverflowAdd(row_envelope_bytes, item_size_bytes, &row_envelope_bytes) ||
        CheckOverflowMul(row_stride_elements, item_size_bytes, &row_stride_bytes) ||
        CheckOverflowMul(column_stride_elements, item_size_bytes, &column_stride_bytes) ||
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
            .column_stride_bytes = column_stride_bytes,
            .item_size_bytes = item_size_bytes,
            .row_count = row_count,
            .column_count = column_count,
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
