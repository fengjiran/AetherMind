#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <limits>
#include <string>

namespace aethermind::cpu::detail {
namespace {

/// @brief Computes the storage bytes required by an arbitrary-rank logical
///        fp32 weight with sign and overflow checks.
StatusOr<size_t> RequiredPackedBytes(std::span<const int64_t> shape,
                                     std::string_view kernel_name) noexcept {
    size_t elements = 1;
    for (const int64_t dimension: shape) {
        if (dimension < 0) {
            return Status::InvalidArgument(std::string(kernel_name) +
                                           " packed shape has a negative dimension");
        }

        if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max()) {
            return Status::Overflow(std::string(kernel_name) +
                                    " packed dimensions exceed size_t");
        }

        if (CheckOverflowMul(elements, static_cast<size_t>(dimension), &elements)) {
            return Status::Overflow(std::string(kernel_name) +
                                    " packed element count overflows size_t");
        }
    }

    size_t bytes = 0;
    if (CheckOverflowMul(elements, sizeof(float), &bytes)) {
        return Status::Overflow(std::string(kernel_name) +
                                " packed byte count overflows size_t");
    }
    return bytes;
}

} // namespace

Status ValidateIdentityPackedWeight(const PackedWeightView& packed,
                                    std::span<const int64_t> expected_shape,
                                    std::string_view kernel_name) noexcept {
    if (packed.logical_dtype != DataType::Float32() ||
        packed.logical_shape.size() != expected_shape.size()) {
        return Status::InvalidArgument(
                std::string(kernel_name) +
                " packed logical metadata does not match the expected weight shape");
    }

    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (packed.logical_shape[i] != expected_shape[i]) {
            return Status::InvalidArgument(
                    std::string(kernel_name) +
                    " packed logical metadata does not match the expected weight shape");
        }
    }

    if (packed.recipe_layout != kCpuIdentityPackingLayout ||
        packed.recipe_alignment != kCpuIdentityPackingAlignment ||
        packed.alignment < kCpuIdentityPackingAlignment) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " requires the cpu_identity packed weight recipe");
    }

    AM_ASSIGN_OR_RETURN(const size_t required_bytes,
                        RequiredPackedBytes(expected_shape, kernel_name));
    if (packed.nbytes < required_bytes) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " packed storage is smaller than its logical weight");
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
