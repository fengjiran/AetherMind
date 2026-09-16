#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <limits>
#include <string>

namespace aethermind::cpu::detail {
namespace {

constexpr std::string_view kCpuIdentityPackingLayout = "cpu_identity";
constexpr size_t kCpuIdentityPackingAlignment = 64;

/// @brief Computes the storage bytes required by a logical `[rows, cols]`
///        fp32 weight with sign and overflow checks.
StatusOr<size_t> RequiredPackedBytes(int64_t total_out_features,
                                     int64_t in_features,
                                     std::string_view kernel_name) noexcept {
    if (total_out_features < 0 || in_features < 0) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " packed shape has a negative dimension");
    }

    if (static_cast<uint64_t>(total_out_features) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(in_features) > std::numeric_limits<size_t>::max()) {
        return Status::Overflow(std::string(kernel_name) +
                                " packed dimensions exceed size_t");
    }

    size_t elements = 0;
    if (CheckOverflowMul(static_cast<size_t>(total_out_features),
                         static_cast<size_t>(in_features), &elements)) {
        return Status::Overflow(std::string(kernel_name) +
                                " packed element count overflows size_t");
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
                                    int64_t total_out_features,
                                    int64_t in_features,
                                    std::string_view kernel_name) noexcept {
    if (packed.logical_dtype != DataType::Float32() ||
        packed.logical_shape.size() != 2U ||
        packed.logical_shape[0] != total_out_features ||
        packed.logical_shape[1] != in_features) {
        return Status::InvalidArgument(
                std::string(kernel_name) +
                " packed logical metadata does not match the expected weight shape");
    }

    if (packed.recipe_layout != kCpuIdentityPackingLayout ||
        packed.recipe_alignment != kCpuIdentityPackingAlignment ||
        packed.alignment < kCpuIdentityPackingAlignment) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " requires the cpu_identity packed weight recipe");
    }

    AM_ASSIGN_OR_RETURN(const size_t required_bytes,
                        RequiredPackedBytes(total_out_features, in_features, kernel_name));
    if (packed.nbytes < required_bytes) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " packed storage is smaller than its logical weight");
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
