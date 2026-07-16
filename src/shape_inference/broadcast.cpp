#include "aethermind/shape_inference/broadcast.h"

#include <algorithm>

namespace aethermind {

StatusOr<std::vector<int64_t>> BroadcastShapes(std::span<const int64_t> lhs_shape,
                                               std::span<const int64_t> rhs_shape) {
    const size_t output_rank = std::max(lhs_shape.size(), rhs_shape.size());
    std::vector<int64_t> output_shape(output_rank, 1);
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

StatusOr<std::vector<int64_t>> BroadcastInputStrides(std::span<const int64_t> input_shape,
                                                     std::span<const int64_t> input_strides,
                                                     std::span<const int64_t> output_shape) {
    if (input_shape.size() != input_strides.size() || input_shape.size() > output_shape.size()) {
        return Status::InvalidArgument("broadcast input metadata rank mismatch");
    }

    std::vector<int64_t> effective_strides(output_shape.size());
    const size_t axis_offset = output_shape.size() - input_shape.size();
    for (size_t output_axis = axis_offset; output_axis < output_shape.size(); ++output_axis) {
        const size_t input_axis = output_axis - axis_offset;
        const int64_t input_dim = input_shape[input_axis];
        const int64_t output_dim = output_shape[output_axis];
        if (input_dim == output_dim) {
            effective_strides[output_axis] = input_strides[input_axis];
        } else if (input_dim == 1) {
            effective_strides[output_axis] = 0;
        } else {
            return Status::InvalidArgument(
                    "input shape is not broadcast-compatible with output shape");
        }
    }
    return effective_strides;
}

StatusOr<SymbolicBroadcastResult> InferBroadcastShape(const SymbolicShape& lhs,
                                                      const SymbolicShape& rhs) {
    if (lhs.IsUnranked() || rhs.IsUnranked()) {
        return Status::InvalidArgument(
                "symbolic broadcast requires both shapes to be ranked");
    }

    const auto lhs_rank = *lhs.rank();
    const auto rhs_rank = *rhs.rank();
    const size_t output_rank = std::max(lhs_rank, rhs_rank);
    const size_t lhs_axis_offset = output_rank - lhs_rank;
    const size_t rhs_axis_offset = output_rank - rhs_rank;

    std::vector<ShapeSymbol> output_shape(output_rank);
    std::vector<DeferredBroadcastAxis> deferred_axes;
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
        if (lhs_dim == rhs_dim && !lhs_dim.IsUnknown()) {
            output_shape[output_axis] = lhs_dim;
            continue;
        }

        // Static 1 yields the other.
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

        // Static N != 1 paired with unresolved (symbolic or unknown)
        // yields N and records a deferred axis.
        const bool lhs_unresolved = lhs_dim.IsSymbolic() || lhs_dim.IsUnknown();
        const bool rhs_unresolved = rhs_dim.IsSymbolic() || rhs_dim.IsUnknown();

        if (lhs_dim.IsStatic() && rhs_unresolved) {
            output_shape[output_axis] = lhs_dim;
            deferred_axes.push_back({
                    .lhs_axis = output_axis - lhs_axis_offset,
                    .rhs_axis = output_axis - rhs_axis_offset,
            });
            continue;
        }

        if (rhs_dim.IsStatic() && lhs_unresolved) {
            output_shape[output_axis] = rhs_dim;
            deferred_axes.push_back({
                    .lhs_axis = output_axis - lhs_axis_offset,
                    .rhs_axis = output_axis - rhs_axis_offset,
            });
            continue;
        }

        output_shape[output_axis] = ShapeSymbol::Unknown();
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
