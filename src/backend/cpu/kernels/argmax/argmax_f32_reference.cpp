#include "argmax_internal.h"

#include <cmath>

namespace aethermind::cpu::detail {
namespace {

/// Selects the winning reduction index of one slice.
///
/// Only a strictly greater candidate replaces the current best, so equal maxima
/// keep the lowest index. A NaN candidate replaces a non-NaN best, and once the
/// best is NaN nothing replaces it: the first NaN of the slice therefore wins,
/// which is deterministic and independent of unordered NaN comparisons.
int64_t ArgmaxF32Slice(const float* slice,
                       int64_t reduction_size,
                       int64_t reduction_stride) noexcept {
    int64_t best_index = 0;
    float best = slice[0];
    for (int64_t r = 1; r < reduction_size; ++r) {
        const float candidate = slice[r * reduction_stride];
        if (std::isnan(candidate)) {
            if (!std::isnan(best)) {
                best = candidate;
                best_index = r;
            }
        } else if (!std::isnan(best) && candidate > best) {
            best = candidate;
            best_index = r;
        }
    }
    return best_index;
}

} // namespace

Status RunArgmaxF32Reference(const ArgmaxF32KernelArgs& args) noexcept {
    for (int64_t flat = 0; flat < args.output_numel; ++flat) {
        int64_t remaining = flat;
        int64_t input_offset = 0;
        int64_t output_offset = 0;
        for (int32_t dim = args.output_rank - 1; dim >= 0; --dim) {
            const int64_t extent = args.output_shape[dim];
            const int64_t coord = remaining % extent;
            remaining /= extent;
            input_offset += coord * args.input_base_strides[dim];
            output_offset += coord * args.output_strides[dim];
        }

        args.output[output_offset] = ArgmaxF32Slice(args.input + input_offset,
                                                    args.reduction_size,
                                                    args.reduction_stride);
    }

    return Status::Ok();
}

} // namespace aethermind::cpu::detail
