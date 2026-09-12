#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
#include "aethermind/base/macros.h"
#include "utils/overflow_check.h"

#include <string>
#include <string_view>

namespace aethermind::cpu::detail {
namespace {

template<typename TensorLike>
Status ValidatePositiveStridesImpl(const TensorLike& tensor,
                                   std::string_view message) noexcept {
    for (int32_t dim = 0; dim < tensor.rank(); ++dim) {
        if (tensor.stride(dim) <= 0) {
            return Status::InvalidArgument(message);
        }
    }
    return Status::Ok();
}

template<typename TensorLike>
Status ValidateFlattenableLeadingAxesImpl(const TensorLike& tensor,
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

StatusOr<int64_t> ComputeRowElementSpan(std::string_view kernel_name,
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

} // namespace

StatusOr<int64_t> ComputeFlattenedRowCount(const TensorView& input,
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

Status ValidatePositiveStrides(const TensorView& tensor,
                               std::string_view message) noexcept {
    return ValidatePositiveStridesImpl(tensor, message);
}

Status ValidatePositiveStrides(const MutableTensorView& tensor,
                               std::string_view message) noexcept {
    return ValidatePositiveStridesImpl(tensor, message);
}

Status ValidateFlattenableLeadingAxes(const TensorView& tensor,
                                      std::string_view kernel_name) noexcept {
    return ValidateFlattenableLeadingAxesImpl(tensor, kernel_name);
}

Status ValidateFlattenableLeadingAxes(const MutableTensorView& tensor,
                                      std::string_view kernel_name) noexcept {
    return ValidateFlattenableLeadingAxesImpl(tensor, kernel_name);
}

Status ValidateRowwiseMaxOffsetRepresentable(std::string_view kernel_name,
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

Status ValidateDisjointOutputRowEnvelopes(std::string_view kernel_name,
                                          int64_t row_count,
                                          int64_t column_count,
                                          int64_t row_stride,
                                          int64_t column_stride) noexcept {
    if (row_count <= 1) {
        return Status::Ok();
    }

    AM_ASSIGN_OR_RETURN(const int64_t row_span,
                        ComputeRowElementSpan(kernel_name, column_count, column_stride));

    if (row_stride < row_span) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " output rows must not overlap");
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
