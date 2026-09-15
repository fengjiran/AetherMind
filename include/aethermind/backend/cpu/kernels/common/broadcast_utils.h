#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_BROADCAST_UTILS_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_BROADCAST_UTILS_H

/// @file broadcast_utils.h
/// @brief Shared broadcast/strided helpers for elementwise CPU kernels.
///
/// Hosts cross-kernel utilities used by the reference broadcast micro-kernels
/// (Add, ElementwiseMul, ...): the canonical max-rank constant, the
/// coordinate-to-offset mapping that applies NumPy-style broadcasting
/// (extent-1 axes reuse their element, leading axes absent from a lower-rank
/// input are pinned implicitly), and the declarations of the binding-time
/// validation core shared by the KernelParamsBuilders. The core is implemented
/// in broadcast_utils.cpp.

#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/status.h"

#include <array>
#include <cstdint>
#include <span>
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

/// @brief Computes the element count of `shape` with overflow and sign checks.
///
/// A zero extent yields 0. Negative extents or products that overflow int64_t
/// are rejected so callers never iterate a negative or wrapped count.
///
/// @param rank Rank of `shape`.
/// @param shape Extents per axis.
/// @return The element count, or InvalidArgument on overflow/negative result.
StatusOr<int64_t> CheckedOutputNumel(int32_t rank,
                                     std::span<const int64_t> shape) noexcept;

/// @brief Compute-ready, dtype-erased args produced by the shared 2-input
///        broadcast validation core.
///
/// This is the canonical shape of a validated elementwise broadcast step:
/// symmetric kernels (Add, ElementwiseMul) project these fields under their
/// own names, and asymmetric kernels (e.g. SiluMul's gate/up) project them
/// onto semantic operand names in their own args struct.
struct ElementwiseBroadcastArgs {
    const void* lhs_data{};
    const void* rhs_data{};
    void* output_data{};
    int64_t numel{};
    bool is_flat{};
    int32_t lhs_rank{};
    int32_t rhs_rank{};
    int32_t output_rank{};
    std::array<int64_t, kMaxRank> lhs_shape{};
    std::array<int64_t, kMaxRank> lhs_strides{};
    std::array<int64_t, kMaxRank> rhs_shape{};
    std::array<int64_t, kMaxRank> rhs_strides{};
    std::array<int64_t, kMaxRank> output_shape{};
    std::array<int64_t, kMaxRank> output_strides{};
};

/// @brief Validates a 2-input broadcast kernel step and produces its
///        compute-ready args.
///
/// Shared builder core for elementwise broadcast kernels (Add,
/// ElementwiseMul, SiluMul, ...). Validates view validity, matching output
/// rank, broadcast compatibility, an overflow-safe element count, and non-null
/// data pointers, then builds the address footprint of each view to require a
/// provably injective output mapping and to reject output/input overlap. Exact
/// in-place against one input is the only accepted aliasing. Finally it fills
/// the returned canonical args, including flat-path eligibility.
///
/// @param context Binding-time per-step views.
/// @param kernel_name Caller name used as the error-message prefix.
/// @param lhs_name Operand name used in errors and footprint labels for the
///                 first input (default "lhs").
/// @param rhs_name Operand name used in errors and footprint labels for the
///                 second input (default "rhs").
/// @return Compute-ready args on success, InvalidArgument on any violated
///         invariant or proven overlap, or Unimplemented when strided layouts
///         leave overlap undecidable.
StatusOr<ElementwiseBroadcastArgs> ValidateAndBuildBroadcastArgs(
        const KernelParamsBuildContext& context,
        std::string_view kernel_name,
        std::string_view lhs_name = "lhs",
        std::string_view rhs_name = "rhs") noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_COMMON_BROADCAST_UTILS_H
