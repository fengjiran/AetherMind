/// Broadcast shape inference and stride expansion utilities.
///
/// Provides two tiers:
/// - Concrete helpers (BroadcastShapes, BroadcastInputStrides) for fully
///   static shapes, used by the constant evaluator and other compile-time
///   passes that already have resolved concrete dimensions.
/// - Symbolic inference (InferBroadcastShape) for shapes that may contain
///   unresolved symbolic or unknown dimensions, used by the graph builder
///   and operator-level shape inference to produce output shapes and deferred
///   axis records before concrete dimensions are known.
#ifndef AETHERMIND_SHAPE_INFERENCE_BROADCAST_H
#define AETHERMIND_SHAPE_INFERENCE_BROADCAST_H

#include "aethermind/base/status.h"
#include "aethermind/shape_inference/shape_symbol.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind {

/// @brief Computes the output shape for broadcasting two concrete shapes.
///
/// Right-aligns the shapes and applies standard NumPy-style broadcasting
/// rules: dimensions must be equal, or one of them must be 1. Rank-0 shapes
/// (empty span) are supported and broadcast as a scalar.
/// @param lhs_shape Left-hand side concrete shape (all dims non-negative).
/// @param rhs_shape Right-hand side concrete shape (all dims non-negative).
/// @return The broadcast output shape, or kInvalidArgument on incompatible dims.
StatusOr<std::vector<int64_t>> BroadcastShapes(std::span<const int64_t> lhs_shape,
                                               std::span<const int64_t> rhs_shape);

/// @brief Expands strides for a broadcast input to match the output rank.
///
/// Leading broadcast axes (where input rank < output rank) receive stride 0
/// so the strided kernel reuses the single element without advancing offset.
/// @param input_shape  Shape of the input tensor.
/// @param input_strides Strides of the input tensor (same rank as input_shape).
/// @param output_shape The broadcast output shape (rank >= input rank).
/// @return Expanded strides matching output_shape rank, or kInvalidArgument.
StatusOr<std::vector<int64_t>> BroadcastInputStrides(std::span<const int64_t> input_shape,
                                                     std::span<const int64_t> input_strides,
                                                     std::span<const int64_t> output_shape);

/// @brief Describes a single output axis produced by broadcasting two
///        non-identical, non-trivial input dimensions.
///
/// lhs_axis and rhs_axis are indices into the original (not right-aligned)
/// input shapes. A deferred axis signals that the broadcast dimension
/// depends on runtime shape information and must be resolved later.
struct DeferredBroadcastAxis {
    size_t lhs_axis;
    size_t rhs_axis;

    // Defaulted to support EXPECT_EQ comparisons in tests and structural
    // equality checks during deferred-axis resolution.
    bool operator==(const DeferredBroadcastAxis&) const noexcept = default;
    bool operator!=(const DeferredBroadcastAxis&) const noexcept = default;
};

/// @brief Result of symbolic broadcast shape inference.
struct SymbolicBroadcastResult {
    /// The broadcast output shape. May contain Unknown() symbols where
    /// the static value could not be resolved.
    SymbolicShape output_shape;
    /// Axes that require deferred resolution. Emitted in ascending
    /// output-axis order.
    std::vector<DeferredBroadcastAxis> deferred_axes;

    bool operator==(const SymbolicBroadcastResult&) const noexcept = default;
    bool operator!=(const SymbolicBroadcastResult&) const noexcept = default;
};

/// @brief Performs broadcast shape inference for two symbolic shapes.
///
/// Rules (deterministic, right-aligned):
/// - Reject unranked shapes.
/// - Identical dimensions are preserved as-is.
/// - A static 1 dimension yields the other dimension.
/// - Incompatible distinct static dimensions produce an error.
/// - A static N != 1 paired with an unresolved dimension yields N and
///   records a deferred axis.
/// - Distinct non-static dimensions yield Unknown() and record a deferred axis.
/// - Missing leading axes (rank expansion) are implicit 1 and never deferred.
///
/// No fresh symbols are created; ambiguous dimensions degrade to Unknown().
/// @param lhs Left-hand side symbolic shape (must be ranked).
/// @param rhs Right-hand side symbolic shape (must be ranked).
/// @return Broadcast result with output shape and deferred axes, or error.
StatusOr<SymbolicBroadcastResult> InferBroadcastShape(const SymbolicShape& lhs,
                                                      const SymbolicShape& rhs);

} // namespace aethermind

#endif // AETHERMIND_SHAPE_INFERENCE_BROADCAST_H
