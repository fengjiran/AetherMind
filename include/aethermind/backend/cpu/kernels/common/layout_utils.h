#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H

/// @file layout_utils.h
/// @brief Shared row-wise layout validation helpers for CPU kernels.
///
/// Hosts the layout checks shared by row-wise kernels (RmsNorm, Linear, ...):
/// collapsing the leading dimensions of an arbitrary-rank view into a
/// [row_count, column_count] row-wise view, and validating that the resulting
/// stride/extent geometry is representable and has disjoint row envelopes.
/// Kernels whose compute treats tensors as row-wise views reference these instead of
/// maintaining private copies.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

#include <cstdint>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Flattens the leading `rank - 1` extents into a row count.
///
/// A zero leading extent yields 0 rows.
///
/// @param input Viewed tensor whose leading extents are multiplied.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return The row count, or InvalidArgument when the product overflows.
/// @pre `input.rank() >= 1`.
StatusOr<int64_t> ComputeFlattenedRowCount(const TensorView& input,
                                           std::string_view kernel_name) noexcept;

/// @brief Verifies that every stride of an immutable view is positive.
///
/// Negative or zero strides break the row-wise offset model; callers pass a
/// kernel-specific message for diagnostics.
Status ValidatePositiveStrides(const TensorView& tensor,
                               std::string_view message) noexcept;

/// @brief Verifies that every stride of a mutable view is positive.
///
/// @see ValidatePositiveStrides(const TensorView&, std::string_view)
Status ValidatePositiveStrides(const MutableTensorView& tensor,
                               std::string_view message) noexcept;

/// @brief Verifies that leading axes can flatten into rows.
///
/// For rank > 2 the row-wise kernel requires `stride(i) == dim(i + 1) *
/// stride(i + 1)` for every leading axis so leading extents fold into a single
/// contiguous row baseline.
///
/// @param tensor View whose leading-axis strides are checked.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return Ok, InvalidArgument on stride-product overflow, or Unimplemented
///         when a leading axis cannot collapse.
/// @pre `tensor.rank() >= 2` (no leading axes to validate otherwise).
Status ValidateFlattenableLeadingAxes(const TensorView& tensor,
                                      std::string_view kernel_name) noexcept;

/// @see ValidateFlattenableLeadingAxes(const TensorView&, std::string_view)
Status ValidateFlattenableLeadingAxes(const MutableTensorView& tensor,
                                      std::string_view kernel_name) noexcept;

/// @brief Verifies that a [row_count, column_count] row-wise view spans a
/// representable max offset.
///
/// Computes `(row_count - 1) * row_stride + (column_count - 1) * column_stride`
/// with overflow checks. This validates arithmetic representability only; it
/// cannot validate allocation bounds because no storage capacity is supplied.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param row_count Number of rows.
/// @param column_count Columns per row.
/// @param row_stride Stride between rows.
/// @param column_stride Stride between columns.
/// @param role Tensor role (e.g. "input") used in the error message.
/// @return Ok when the max offset is representable, otherwise InvalidArgument.
Status ValidateRowwiseMaxOffsetRepresentable(std::string_view kernel_name,
                                             int64_t row_count,
                                             int64_t column_count,
                                             int64_t row_stride,
                                             int64_t column_stride,
                                             std::string_view role) noexcept;

/// @brief Verifies that distinct output row envelopes do not overlap.
///
/// This is a conservative supported-layout constraint: column-stride holes are
/// part of each row envelope even though they are not logical elements. Row
/// envelopes are disjoint when `row_stride >= row element span`, or when there
/// is at most one row.
Status ValidateDisjointOutputRowEnvelopes(std::string_view kernel_name,
                                          int64_t row_count,
                                          int64_t column_count,
                                          int64_t row_stride,
                                          int64_t column_stride) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H
