#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_BROADCAST_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_BROADCAST_UTILS_H

/// @file broadcast_utils.h
/// @brief Shared broadcast/strided helpers for elementwise CPU kernels.
///
/// Hosts cross-kernel utilities used by the reference broadcast micro-kernels
/// (Add, ElementwiseMul, ...): the canonical max-rank constant, the
/// coordinate-to-offset mapping that applies NumPy-style broadcasting
/// (extent-1 axes reuse their element, leading axes absent from a lower-rank
/// input are pinned implicitly), and the binding-time validation core shared
/// by the KernelParamsBuilders.

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/status.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace aethermind::cpu::detail {

/// @brief Maximum tensor rank accepted by CPU kernel args structs.
///
/// Single source for the fixed-size shape/stride arrays in compute-ready
/// args; kernels reference this instead of maintaining private copies.
inline constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

/// @brief Maps an output coordinate to a linear offset in a possibly
/// broadcast (or lower-rank) input tensor.
///
/// Broadcast axes (extent 1) pin their coordinate to 0 so the single element
/// is reused; leading axes absent from a lower-rank input are skipped via
/// axis_offset.
///
/// @param input_shape Shape of the input tensor being mapped.
/// @param output_rank Rank of the broadcast output.
/// @param input_strides Input strides in elements.
/// @param out_coord Decomposed output coordinate; valid for every axis in
///        `[0, output_rank)`.
/// @return Linear offset into the input data for `out_coord`.
/// @pre `input_shape.size() + axis_offset == output_rank`.
inline int64_t MapCoordToOffset(std::span<const int64_t> input_shape,
                                int32_t output_rank,
                                std::span<const int64_t> input_strides,
                                const std::array<int64_t, kMaxRank>& out_coord) noexcept {
    const auto input_rank = static_cast<int32_t>(input_shape.size());
    const int32_t axis_offset = output_rank - input_rank;
    int64_t offset = 0;
    for (int32_t axis = axis_offset; axis < output_rank; ++axis) {
        const int32_t input_axis = axis - axis_offset;
        const int64_t dim = input_shape[input_axis];
        const int64_t idx = dim == 1 ? int64_t{0} : out_coord[axis];
        offset += idx * input_strides[input_axis];
    }
    return offset;
}

/// @brief Checks that `lhs_shape` and `rhs_shape` broadcast to `output_shape`.
///
/// Negative dimensions are rejected. The output dimension must equal the
/// broadcast result for every axis: lhs==1 → rhs; rhs==1 or equal → lhs.
///
/// @param lhs_shape LHS shape in elements per axis.
/// @param rhs_shape RHS shape in elements per axis.
/// @param output_shape Result shape being validated.
/// @return True when the three shapes are broadcast-compatible.
inline bool ValidateBroadcastCompatible(std::span<const int64_t> lhs_shape,
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

/// @brief Verifies that `(shape, strides)` span a representable max offset.
///
/// Computes `sum((shape[i] - 1) * strides[i])` with overflow checks; a zero
/// extent short-circuits to Ok. Catches views whose declared shape would
/// index past the end of their storage when traversed with their strides.
///
/// @param kernel_name Caller name used as the error-message prefix.
/// @param shape Extents per axis.
/// @param strides Strides in elements per axis.
/// @param name Tensor role (e.g. "lhs") used in the error message.
/// @return Ok when the max offset is representable, otherwise InvalidArgument.
inline Status ValidateMaxOffset(std::string_view kernel_name,
                                std::span<const int64_t> shape,
                                std::span<const int64_t> strides,
                                std::string_view name) noexcept {
    const auto rank = static_cast<int32_t>(shape.size());
    if (rank == 0) {
        return Status::Ok();
    }

    int64_t max_offset = 0;
    for (int32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return Status::Ok();
        }

        int64_t contrib = 0;
        if (CheckOverflowMul(shape[i] - 1, strides[i], &contrib)) {
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " +
                    std::string(name) + " offset overflow");
        }

        int64_t new_max = 0;
        if (CheckOverflowAdd(max_offset, contrib, &new_max)) {
            return Status::InvalidArgument(
                    std::string(kernel_name) + " " +
                    std::string(name) + " offset overflow");
        }
        max_offset = new_max;
    }
    return Status::Ok();
}

/// @brief Computes the element count of `shape` with overflow and sign checks.
///
/// A zero extent yields 0. Negative extents or products that overflow int64_t
/// are rejected so callers never iterate a negative or wrapped count.
///
/// @param rank Rank of `shape`.
/// @param shape Extents per axis.
/// @return The element count, or InvalidArgument on overflow/negative result.
inline StatusOr<int64_t> CheckedOutputNumel(int32_t rank,
                                            std::span<const int64_t> shape) noexcept {
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

/// @brief Validates a 2-input broadcast kernel step and populates its
/// compute-ready args.
///
/// Shared builder core for elementwise broadcast kernels (Add,
/// ElementwiseMul, ...). Validates view validity, matching output rank,
/// broadcast compatibility, an overflow-safe element count and max offset,
/// and non-null data pointers, then fills the rank/shape/stride fields of
/// the kernel args. Kernels whose args carry an `is_flat` member additionally
/// get the flat-path eligibility computed here.
///
/// @param context Binding-time per-step views.
/// @param args Compute-ready args being populated.
/// @param kernel_name Caller name used as the error-message prefix.
/// @return Ok on success, InvalidArgument on any violated invariant.
template<typename KernelArgs>
Status ValidateAndBuildElementwiseArgs(const KernelParamsBuildContext& context,
                                       KernelArgs& args,
                                       std::string_view kernel_name) noexcept {
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
                std::string(kernel_name) + " requires a valid lhs TensorView");
    }

    if (!rhs.is_valid()) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires a valid rhs TensorView");
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
        return Status::InvalidArgument(
                std::string(kernel_name) + " output rank exceeds maximum supported rank");
    }

    if (!ValidateBroadcastCompatible(lhs.shape(), rhs.shape(), output.shape())) {
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
        args = KernelArgs{};
        return Status::Ok();
    }

    if (lhs.data() == nullptr) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires non-null lhs data");
    }

    if (rhs.data() == nullptr) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires non-null rhs data");
    }

    if (output.data() == nullptr) {
        return Status::InvalidArgument(
                std::string(kernel_name) + " requires non-null output data");
    }

    {
        auto status = ValidateMaxOffset(kernel_name, lhs.shape(), lhs.strides(), "lhs");
        if (!status.ok()) {
            return status;
        }

        status = ValidateMaxOffset(kernel_name, rhs.shape(), rhs.strides(), "rhs");
        if (!status.ok()) {
            return status;
        }

        status = ValidateMaxOffset(kernel_name, output.shape(), output.strides(), "output");
        if (!status.ok()) {
            return status;
        }
    }

    args.lhs_data = static_cast<decltype(args.lhs_data)>(lhs.data());
    args.rhs_data = static_cast<decltype(args.rhs_data)>(rhs.data());
    args.output_data = static_cast<decltype(args.output_data)>(output.data());
    args.numel = numel;

    // Determine flat-path eligibility when the args model supports it.
    if constexpr (requires { args.is_flat; }) {
        args.is_flat = lhs.is_contiguous() && rhs.is_contiguous() &&
                       output.is_contiguous() && lhs.shape() == output.shape() &&
                       rhs.shape() == output.shape();
    }

    // Populate broadcast / strided path metadata.
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

    return Status::Ok();
}

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_BROADCAST_UTILS_H