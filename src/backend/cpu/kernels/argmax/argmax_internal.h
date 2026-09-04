#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ARGMAX_ARGMAX_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ARGMAX_ARGMAX_INTERNAL_H

/// @file argmax_internal.h
/// @brief Internal declarations for the CPU FP32 reference ArgMax kernel.
///
/// Defines the pre-validated compute-ready args (`ArgmaxF32KernelArgs`) and the
/// scalar reference implementation. Axis normalization, dtype dispatch, and the
/// input-to-output shape mapping all happen in the binding-time params builder,
/// so the prepared args are self-contained and the micro-kernel never re-reads
/// `OpParams` or the borrowed `TensorView`s.

#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/status.h"

#include <array>
#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for the ArgMax reference micro-kernel.
///
/// The reduction axis is already folded into `reduction_size`/`reduction_stride`
/// and removed from the per-axis arrays: all three arrays are indexed by output
/// axis, so `input_base_strides[d]` is the input stride of the axis that precedes
/// output axis `d` in the input. Unused slots (index >= `output_rank`) stay zero.
///
/// Invariants established by the params builder:
/// - `output_numel == 0` means no work: pointers may be null and strides are
///   unvalidated, and the micro-kernel dereferences nothing.
/// - Otherwise `reduction_size >= 1`, every `output_shape[d] >= 1`, all strides
///   are positive, and every reachable offset is representable in int64_t.
struct ArgmaxF32KernelArgs {
    const float* input{};
    int64_t* output{};

    int64_t output_numel{};
    int64_t reduction_size{};
    int64_t reduction_stride{};

    int32_t output_rank{};

    std::array<int64_t, ShapeAndStride::kMaxRank> output_shape{};
    std::array<int64_t, ShapeAndStride::kMaxRank> input_base_strides{};
    std::array<int64_t, ShapeAndStride::kMaxRank> output_strides{};
};

/// @brief Runs the scalar FP32 ArgMax reference micro-kernel.
///
/// Writes, for every logical output element, the lowest index of the maximum
/// value along the reduction slice. The first NaN of a slice wins and is never
/// replaced, which keeps the lowest-index tie rule deterministic.
///
/// @param args Pre-validated kernel arguments.
/// @return Ok; no argument validation is repeated here.
Status RunArgmaxF32Reference(const ArgmaxF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ARGMAX_ARGMAX_INTERNAL_H
