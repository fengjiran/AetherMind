#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H

/// @file layout_utils.h
/// @brief Shared row-wise layout validation helpers for CPU kernels.
///
/// Hosts pure metadata checks shared by row-wise kernels: collapsing leading
/// dimensions of an arbitrary-rank view and validating positive strides. The
/// checked element/address geometry and mutable-output policy belong to the
/// complete row-wise analysis in alias_utils.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"

#include <cstdint>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Flattens the leading `rank - 1` extents into a row count.
///
/// A zero leading extent yields 0 rows.
///
/// @param tensor Viewed tensor whose leading extents are multiplied.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return The row count, or InvalidArgument when the product overflows.
/// @pre `tensor.rank() >= 1`.
StatusOr<int64_t> ComputeFlattenedRowCount(const TensorView& tensor,
                                           std::string_view kernel_name) noexcept;

/// @see ComputeFlattenedRowCount(const TensorView&, std::string_view)
StatusOr<int64_t> ComputeFlattenedRowCount(const MutableTensorView& tensor,
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

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_LAYOUT_UTILS_H
