// Broadcast shape inference and stride expansion.
//
// Implements the concrete and symbolic tiers declared in broadcast.h. The
// concrete helpers (BroadcastShapes, BroadcastInputStrides) assume fully
// resolved dimensions; InferBroadcastShape handles unresolved symbolic or
// unknown dimensions and emits deferred-axis records. All tiers use
// right-aligned NumPy-style broadcasting.
#include "aethermind/shape_inference/broadcast.h"

#include <algorithm>
#include <cstddef>

namespace aethermind {

AM_NODISCARD StatusOr<std::vector<int64_t>> BroadcastShapes(std::span<const int64_t> lhs_shape,
                                                            std::span<const int64_t> rhs_shape) {
    const size_t output_rank = std::max(lhs_shape.size(), rhs_shape.size());
    std::vector<int64_t> output_shape(output_rank, 1);
    // Right-align the inputs by padding leading axes with 1. The offset maps a
    // right-aligned output axis back to the original (left-aligned) input index.
    const size_t lhs_axis_offset = output_rank - lhs_shape.size();
    const size_t rhs_axis_offset = output_rank - rhs_shape.size();
    for (size_t output_axis = 0; output_axis < output_rank; ++output_axis) {
        const int64_t lhs_dim = output_axis < lhs_axis_offset
                                        ? 1
                                        : lhs_shape[output_axis - lhs_axis_offset];
        const int64_t rhs_dim = output_axis < rhs_axis_offset
                                        ? 1
                                        : rhs_shape[output_axis - rhs_axis_offset];
        if (lhs_dim < 0 || rhs_dim < 0) {
            return Status::InvalidArgument("broadcast dimensions must be non-negative");
        }

        if (lhs_dim == 1) {
            output_shape[output_axis] = rhs_dim;
        } else if (rhs_dim == 1 || lhs_dim == rhs_dim) {
            output_shape[output_axis] = lhs_dim;
        } else {
            return Status::InvalidArgument("broadcast dimensions are incompatible");
        }
    }
    return output_shape;
}

AM_NODISCARD StatusOr<std::vector<int64_t>> BroadcastInputStrides(std::span<const int64_t> input_shape,
                                                                  std::span<const int64_t> input_strides,
                                                                  std::span<const int64_t> output_shape) {
    if (input_shape.size() != input_strides.size() || input_shape.size() > output_shape.size()) {
        return Status::InvalidArgument("broadcast input metadata rank mismatch");
    }

    // Leading axes (output_axis < axis_offset) correspond to input rank
    // expansion and reuse the single input element, so their stride is 0.
    // Value-initialization below zeros all entries, so we only need to
    // fill in the trailing axes that align with the input.
    std::vector<int64_t> effective_strides(output_shape.size());
    const size_t axis_offset = output_shape.size() - input_shape.size();
    for (size_t output_axis = axis_offset; output_axis < output_shape.size(); ++output_axis) {
        const size_t input_axis = output_axis - axis_offset;
        const int64_t input_dim = input_shape[input_axis];
        const int64_t output_dim = output_shape[output_axis];
        if (input_dim < 0 || output_dim < 0) {
            return Status::InvalidArgument("broadcast dimensions must be non-negative");
        }

        if (input_dim == output_dim) {
            effective_strides[output_axis] = input_strides[input_axis];
        } else if (input_dim == 1) {
            // A broadcast (size-1) axis is reused across the output extent, so
            // its stride is 0: the strided kernel reads the same element
            // repeatedly without advancing the offset.
            effective_strides[output_axis] = 0;
        } else {
            return Status::InvalidArgument(
                    "input shape is not broadcast-compatible with output shape");
        }
    }
    return effective_strides;
}

AM_NODISCARD StatusOr<SymbolicBroadcastResult> InferBroadcastShape(const SymbolicShape& lhs,
                                                                   const SymbolicShape& rhs) {
    if (lhs.IsUnranked() || rhs.IsUnranked()) {
        return Status::InvalidArgument(
                "symbolic broadcast requires both shapes to be ranked");
    }

    const auto lhs_rank = *lhs.rank();
    const auto rhs_rank = *rhs.rank();
    const size_t output_rank = std::max(lhs_rank, rhs_rank);
    // Right-align the inputs: leading axes beyond an input's rank are implicit
    // size-1 axes and never produce a deferred record.
    const size_t lhs_axis_offset = output_rank - lhs_rank;
    const size_t rhs_axis_offset = output_rank - rhs_rank;

    std::vector<ShapeSymbol> output_shape(output_rank);
    std::vector<DeferredBroadcastAxis> deferred_axes;
    // At most one deferred axis per output dimension.
    deferred_axes.reserve(output_rank);

    for (size_t output_axis = 0; output_axis < output_rank; ++output_axis) {
        const bool lhs_implicit = output_axis < lhs_axis_offset;
        const bool rhs_implicit = output_axis < rhs_axis_offset;

        if (lhs_implicit) {
            output_shape[output_axis] = rhs[output_axis - rhs_axis_offset];
            continue;
        }

        if (rhs_implicit) {
            output_shape[output_axis] = lhs[output_axis - lhs_axis_offset];
            continue;
        }

        // Both axes are explicit.
        const ShapeSymbol& lhs_dim = lhs[output_axis - lhs_axis_offset];
        const ShapeSymbol& rhs_dim = rhs[output_axis - rhs_axis_offset];

        // Identical dimensions are preserved. Two Unknown sentinels are
        // NOT proven identical, so they fall through to the deferred path.
        // Static zero dimensions also match here (0 == 0), which is the
        // correct NumPy broadcast result for equal-zero axes.
        if (lhs_dim == rhs_dim && !lhs_dim.IsUnknown()) {
            output_shape[output_axis] = lhs_dim;
            continue;
        }

        // Static 1 yields the other. Static 0 paired with 1 falls through
        // to the incompatible-static check below (0 != 1), which is correct.
        if (lhs_dim.IsStatic() && lhs_dim.GetStaticValue() == 1) {
            output_shape[output_axis] = rhs_dim;
            continue;
        }

        if (rhs_dim.IsStatic() && rhs_dim.GetStaticValue() == 1) {
            output_shape[output_axis] = lhs_dim;
            continue;
        }

        // Incompatible distinct static dimensions.
        if (lhs_dim.IsStatic() && rhs_dim.IsStatic()) {
            return Status::InvalidArgument(
                    "broadcast dimensions are incompatible");
        }

        // Static N != 1 paired with an unresolved (symbolic or unknown)
        // dimension yields N and records a deferred axis to be checked at
        // runtime, since the unresolved side may yet equal N.
        const bool lhs_unresolved = lhs_dim.IsSymbolic() || lhs_dim.IsUnknown();
        const bool rhs_unresolved = rhs_dim.IsSymbolic() || rhs_dim.IsUnknown();

        if (lhs_dim.IsStatic() && rhs_unresolved) {
            output_shape[output_axis] = lhs_dim;
        } else if (rhs_dim.IsStatic() && lhs_unresolved) {
            output_shape[output_axis] = rhs_dim;
        } else {
            // Neither side is statically resolvable: degrade to Unknown.
            output_shape[output_axis] = ShapeSymbol::Unknown();
        }

        deferred_axes.push_back({
                .lhs_axis = output_axis - lhs_axis_offset,
                .rhs_axis = output_axis - rhs_axis_offset,
        });
    }

    return SymbolicBroadcastResult{
            .output_shape = SymbolicShape(std::move(output_shape)),
            .deferred_axes = std::move(deferred_axes),
    };
}

}// namespace aethermind
