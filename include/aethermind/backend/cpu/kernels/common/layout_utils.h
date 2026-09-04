#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H

/// @file layout_utils.h
/// @brief Shared row-wise layout validation helpers for CPU kernels.
///
/// Hosts the layout checks shared by row-wise kernels (RmsNorm, Linear, ...):
/// collapsing the leading dimensions of an arbitrary-rank view into a
/// [row_count, column_count] row-wise view, and validating that the resulting
/// stride/extent geometry is representable and non-overlapping. Kernels whose
/// compute treats tensors as row-wise views reference these instead of
/// maintaining private copies.

#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "utils/overflow_check.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Multiplies the leading `rank - 1` extents into an element row count.
///
/// A zero leading extent yields 0 rows.
///
/// @param input Viewed tensor whose leading extents are multiplied.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return The row count, or InvalidArgument when the product overflows.
/// @pre `input.rank() >= 1`.
inline StatusOr<int64_t> ComputeRowCount(const TensorView& input,
                                         std::string_view kernel_name) noexcept {
    int64_t row_count = 1;
    for (int32_t i = 0; i < input.rank() - 1; ++i) {
        const int64_t extent = input.dim(i);
        if (extent == 0) {
            return int64_t{0};
        }

        int64_t next_row_count = 0;
        if (CheckOverflowMul(row_count, extent, &next_row_count)) {
            return Status::InvalidArgument(
                    std::string(kernel_name) + " row count overflow");
        }
        row_count = next_row_count;
    }
    return row_count;
}

/// @brief Verifies that every stride of an immutable view is positive.
///
/// Negative or zero strides break the row-wise offset model; callers pass a
/// kernel-specific message for diagnostics.
inline Status ValidatePositiveStrides(const TensorView& tensor,
                                      std::string_view message) noexcept {
    for (int32_t dim = 0; dim < tensor.rank(); ++dim) {
        if (tensor.stride(dim) <= 0) {
            return Status::InvalidArgument(message);
        }
    }
    return Status::Ok();
}

/// @brief Verifies that every stride of a mutable view is positive.
///
/// @see ValidatePositiveStrides(const TensorView&, std::string_view)
inline Status ValidatePositiveStrides(const MutableTensorView& tensor,
                                      std::string_view message) noexcept {
    for (int32_t i = 0; i < tensor.rank(); ++i) {
        if (tensor.stride(i) <= 0) {
            return Status::InvalidArgument(message);
        }
    }
    return Status::Ok();
}

/// @brief Verifies that leading dimensions can collapse into rows.
///
/// For rank > 2 the row-wise kernel requires `stride(i) == dim(i + 1) *
/// stride(i + 1)` for every leading axis so leading extents fold into a single
/// contiguous row baseline.
///
/// @param tensor View whose leading axes are checked.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return Ok, InvalidArgument on stride-product overflow, or Unimplemented
///         when a leading axis cannot collapse.
/// @pre `tensor.rank() >= 2` (no leading axes to validate otherwise).
template<typename TensorLike>
Status ValidateCollapsibleLeadingDimensions(const TensorLike& tensor,
                                            std::string_view kernel_name) noexcept {
    for (int32_t i = 0; i < tensor.rank() - 2; ++i) {
        int64_t expected_stride = 0;
        if (CheckOverflowMul(tensor.dim(i + 1), tensor.stride(i + 1), &expected_stride)) {
            return Status::InvalidArgument(std::string(kernel_name) +
                                           " leading-dimension stride product overflow");
        }

        if (tensor.stride(i) != expected_stride) {
            return Status::Unimplemented(std::string(kernel_name) +
                                         " requires collapsible leading dimensions for rank > 2");
        }
    }
    return Status::Ok();
}

/// @brief Verifies that a [row_count, column_count] row-wise view spans a
/// representable max offset.
///
/// Computes `(row_count - 1) * row_stride + (column_count - 1) * column_stride`
/// with overflow checks, catching views whose declared extents would index past
/// the end of their storage.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param row_count Number of rows.
/// @param column_count Columns per row.
/// @param row_stride Stride between rows.
/// @param column_stride Stride between columns.
/// @param role Tensor role (e.g. "input") used in the error message.
/// @return Ok when the max offset is representable, otherwise InvalidArgument.
inline Status ValidateRowColMaxOffset(std::string_view kernel_name,
                                      int64_t row_count,
                                      int64_t column_count,
                                      int64_t row_stride,
                                      int64_t column_stride,
                                      std::string_view role) noexcept {
    int64_t row_offset = 0;
    if (CheckOverflowMul(row_count - 1, row_stride, &row_offset)) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " " + std::string(role) + " offset overflow");
    }

    int64_t column_offset = 0;
    if (CheckOverflowMul(column_count - 1, column_stride, &column_offset)) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " " + std::string(role) + " offset overflow");
    }

    int64_t max_offset = 0;
    if (CheckOverflowAdd(row_offset, column_offset, &max_offset)) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " " + std::string(role) + " offset overflow");
    }
    return Status::Ok();
}

/// @brief Computes the element span of one row in a column-strided view.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param column_count Columns per row.
/// @param column_stride Stride between columns.
/// @return The row span, or InvalidArgument on overflow.
inline StatusOr<int64_t> ComputeRowSpan(std::string_view kernel_name,
                                        int64_t column_count,
                                        int64_t column_stride) noexcept {
    int64_t last_column_offset = 0;
    if (CheckOverflowMul(column_count - 1, column_stride, &last_column_offset)) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " output row span overflow");
    }

    int64_t row_span = 0;
    if (CheckOverflowAdd(last_column_offset, int64_t{1}, &row_span)) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " output row span overflow");
    }
    return row_span;
}

/// @brief Verifies that distinct output rows do not overlap in memory.
///
/// Rows that overlap would make the kernel overwrite its own output as it
/// walks rows; safe only when `row_stride >= row span` or there is at most one
/// row.
inline Status ValidateNonOverlappingOutputRows(std::string_view kernel_name,
                                               int64_t row_count,
                                               int64_t column_count,
                                               int64_t row_stride,
                                               int64_t column_stride) noexcept {
    if (row_count <= 1) {
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_span, ComputeRowSpan(kernel_name, column_count, column_stride));

    if (row_stride < row_span) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " output rows must not overlap");
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H