#include "aethermind/backend/cpu/kernels/common/layout_utils.h"
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

template<typename TensorLike>
StatusOr<int64_t> ComputeFlattenedRowCountImpl(const TensorLike& tensor,
                                               std::string_view kernel_name) noexcept {
    int64_t row_count = 1;
    for (int32_t i = 0; i < tensor.rank() - 1; ++i) {
        const int64_t extent = tensor.dim(i);
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

} // namespace

StatusOr<int64_t> ComputeFlattenedRowCount(const TensorView& tensor,
                                           std::string_view kernel_name) noexcept {
    return ComputeFlattenedRowCountImpl(tensor, kernel_name);
}

StatusOr<int64_t> ComputeFlattenedRowCount(const MutableTensorView& tensor,
                                           std::string_view kernel_name) noexcept {
    return ComputeFlattenedRowCountImpl(tensor, kernel_name);
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

} // namespace aethermind::cpu::detail
