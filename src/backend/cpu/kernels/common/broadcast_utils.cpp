#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <span>
#include <string>

namespace aethermind::cpu::detail {
namespace {

/// @brief Checks that `lhs_shape` and `rhs_shape` broadcast to `output_shape`.
///
/// Negative dimensions are rejected. The output dimension must equal the
/// broadcast result for every axis: lhs==1 → rhs; rhs==1 or equal → lhs.
///
/// @param lhs_shape LHS shape in elements per axis.
/// @param rhs_shape RHS shape in elements per axis.
/// @param output_shape Result shape being validated.
/// @return True when the three shapes are broadcast-compatible.
bool ValidateBroadcastCompatible(std::span<const int64_t> lhs_shape,
                                 std::span<const int64_t> rhs_shape,
                                 std::span<const int64_t> output_shape) noexcept {
    const auto output_rank = static_cast<int32_t>(output_shape.size());
    const auto lhs_offset = output_rank - static_cast<int32_t>(lhs_shape.size());
    const auto rhs_offset = output_rank - static_cast<int32_t>(rhs_shape.size());

    for (int32_t axis = 0; axis < output_rank; ++axis) {
        const int64_t out_dim = output_shape[axis];
        const int64_t lhs_dim = axis < lhs_offset ? 1 : lhs_shape[axis - lhs_offset];
        const int64_t rhs_dim = axis < rhs_offset ? 1 : rhs_shape[axis - rhs_offset];

        if (lhs_dim < 0 || rhs_dim < 0) {
            return false;
        }

        // Broadcast rule: lhs==1 → rhs; rhs==1 or equal → lhs.
        const int64_t expected = lhs_dim == 1                         ? rhs_dim
                                 : rhs_dim == 1 || lhs_dim == rhs_dim ? lhs_dim
                                                                      : int64_t{-1};
        if (expected < 0 || out_dim != expected) {
            return false;
        }
    }
    return true;
}

} // namespace

StatusOr<int64_t> CheckedOutputNumel(int32_t rank, std::span<const int64_t> shape) noexcept {
    if (rank == 0) {
        return int64_t{1};
    }

    int64_t count = 1;
    for (int32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return int64_t{0};
        }

        int64_t next = 0;
        if (CheckOverflowMul(count, shape[i], &next)) {
            return Status::InvalidArgument(
                    "kernel output element count overflow");
        }

        if (next < 0) {
            return Status::InvalidArgument(
                    "kernel output element count exceeds int64_t");
        }
        count = next;
    }
    return count;
}

StatusOr<ElementwiseBroadcastArgs> ValidateAndBuildBroadcastArgs(
        const KernelParamsBuildContext& context,
        std::string_view kernel_name,
        std::string_view lhs_name,
        std::string_view rhs_name) noexcept {
    const auto inputs = context.inputs;
    const auto outputs = context.outputs;
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires 2 inputs and 1 output");
    }

    const TensorView& lhs = inputs[0];
    const TensorView& rhs = inputs[1];
    const MutableTensorView& output = outputs[0];

    if (!lhs.is_valid()) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires a valid " + std::string(lhs_name) +
                " TensorView");
    }

    if (!rhs.is_valid()) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires a valid " + std::string(rhs_name) +
                " TensorView");
    }

    if (!output.is_valid()) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires a valid output MutableTensorView");
    }

    const int32_t output_rank = output.rank();
    const int32_t expected_rank = std::max(lhs.rank(), rhs.rank());
    if (output_rank != expected_rank) {
        return Status::InvalidArgument(
                std::string(kernel_name) +
                " output rank must equal max(lhs rank, rhs rank)");
    }

    if (output_rank > static_cast<int32_t>(kMaxRank)) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " output rank exceeds maximum supported rank");
    }

    if (!ValidateBroadcastCompatible(lhs.shape(), rhs.shape(),
                                     output.shape())) {
        return Status::InvalidArgument(
                std::string(kernel_name) +
                " input shapes are not broadcast-compatible with output shape");
    }

    const auto numel_or = CheckedOutputNumel(output_rank, output.shape());
    if (!numel_or.ok()) {
        return numel_or.status();
    }

    const int64_t numel = numel_or.value();
    if (numel == 0) {
        return ElementwiseBroadcastArgs{};
    }

    if (lhs.data() == nullptr) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " requires non-null " + std::string(lhs_name) + " data");
    }

    if (rhs.data() == nullptr) {
        return Status::InvalidArgument(std::string(kernel_name) +
                                       " requires non-null " + std::string(rhs_name) + " data");
    }

    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires non-null output data");
    }

    AM_ASSIGN_OR_RETURN(
            const StridedAddressFootprint lhs_footprint,
            BuildStridedAddressFootprint(
                    lhs, std::string(kernel_name) + " " + std::string(lhs_name)));
    AM_ASSIGN_OR_RETURN(
            const StridedAddressFootprint rhs_footprint,
            BuildStridedAddressFootprint(
                    rhs, std::string(kernel_name) + " " + std::string(rhs_name)));
    AM_ASSIGN_OR_RETURN(const StridedAddressFootprint output_footprint,
                        BuildStridedAddressFootprint(output, std::string(kernel_name) +
                                                                     " output"));

    // A non-injective output writes one slot from several coordinates, so the
    // stored value would depend on iteration order.
    AM_RETURN_IF_ERROR(ValidateLayoutInjectivity(
            kernel_name, output_footprint.injectivity(), "output"));

    // Exact in-place against one input is safe: that element is read immediately
    // before its own slot is written. Identical mapping implies an equal shape,
    // so a broadcast input (an extent-1 element reused by every output element)
    // never qualifies here and stays subject to the overlap check.
    if (!HaveIdenticalViewMapping(lhs, output)) {
        AM_RETURN_IF_ERROR(ValidateStridedDisjoint(
                kernel_name, output_footprint, "output",
                lhs_footprint, lhs_name));
    }

    if (!HaveIdenticalViewMapping(rhs, output)) {
        AM_RETURN_IF_ERROR(ValidateStridedDisjoint(
                kernel_name, output_footprint, "output",
                rhs_footprint, rhs_name));
    }

    ElementwiseBroadcastArgs args{};
    args.lhs_data = lhs.data();
    args.rhs_data = rhs.data();
    args.output_data = output.data();
    args.numel = numel;
    args.is_flat = lhs.is_contiguous() && rhs.is_contiguous() &&
                   output.is_contiguous() && lhs.shape() == output.shape() &&
                   rhs.shape() == output.shape();
    args.lhs_rank = lhs.rank();
    args.rhs_rank = rhs.rank();
    args.output_rank = output_rank;
    for (int32_t i = 0; i < lhs.rank(); ++i) {
        args.lhs_shape[i] = lhs.shape()[i];
        args.lhs_strides[i] = lhs.strides()[i];
    }

    for (int32_t i = 0; i < rhs.rank(); ++i) {
        args.rhs_shape[i] = rhs.shape()[i];
        args.rhs_strides[i] = rhs.strides()[i];
    }

    for (int32_t i = 0; i < output_rank; ++i) {
        args.output_shape[i] = output.shape()[i];
        args.output_strides[i] = output.strides()[i];
    }

    return args;
}

} // namespace aethermind::cpu::detail
