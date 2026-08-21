#include "aethermind/shape_inference/shape_constraint_evaluator.h"
#include "container/array_view.h"
#include "utils/logging.h"
#include "utils/overflow_check.h"
#include "utils/variant_utils.h"

#include <string>

namespace aethermind {
namespace {

AM_NODISCARD const SymbolicShape* ResolveShape(const TensorPort& port,
                                               const std::span<const SymbolicShape> inputs,
                                               const std::span<const SymbolicShape> outputs) {
    if (port.direction == TensorPortType::kInput) {
        AM_CHECK(port.tensor_idx < inputs.size(),
                 "Shape constraint references missing input tensor %zu", port.tensor_idx);
        return &inputs[port.tensor_idx];
    }

    AM_CHECK(port.tensor_idx < outputs.size(),
             "Shape constraint references missing output tensor %zu", port.tensor_idx);
    return &outputs[port.tensor_idx];
}

/// @brief Resolves the runtime shape referenced by a tensor port.
///
/// Returns kInvalidArgument when the port direction is invalid or the tensor
/// index falls outside the bound spans. The runtime hot path must never abort
/// on a malformed constraint; plan-build validation (ExecutionPlan::Create)
/// guarantees references are in range for compiled plans.
StatusOr<IntArrayView> ResolveRuntimeShape(
        const TensorPort& port,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    switch (port.direction) {
        case TensorPortType::kInput:
            if (port.tensor_idx >= inputs.size()) {
                return Status::InvalidArgument(
                        "Shape constraint references missing input tensor " +
                        std::to_string(port.tensor_idx));
            }
            return inputs[port.tensor_idx].shape();

        case TensorPortType::kOutput:
            if (port.tensor_idx >= outputs.size()) {
                return Status::InvalidArgument(
                        "Shape constraint references missing output tensor " +
                        std::to_string(port.tensor_idx));
            }
            return outputs[port.tensor_idx].shape();
    }
    return Status::InvalidArgument(
            "Shape constraint references a tensor port with an invalid direction");
}

AM_NODISCARD std::optional<ShapeSymbol> ResolveSymbolicDim(
        const DimLocator& locator,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto* shape = ResolveShape(locator.tensor_port, inputs, outputs);
    const auto rank = shape->rank();
    if (!rank.has_value()) {
        return std::nullopt;
    }

    AM_CHECK(locator.dim_index < *rank,
             "Shape constraint references missing dimension %zu", locator.dim_index);
    return (*shape)[locator.dim_index];
}

StatusOr<int64_t> ResolveRuntimeDim(
        const DimLocator& locator,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto shape = ResolveRuntimeShape(locator.tensor_port, inputs, outputs);
    if (!shape.ok()) {
        return shape.status();
    }

    if (locator.dim_index >= shape->size()) {
        return Status::InvalidArgument(
                "Shape constraint references missing dimension " +
                std::to_string(locator.dim_index));
    }
    return (*shape)[locator.dim_index];
}

AM_NODISCARD std::optional<size_t> ResolveSymbolicRank(
        const TensorPort& port,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    return ResolveShape(port, inputs, outputs)->rank();
}

StatusOr<size_t> ResolveRuntimeRank(
        const TensorPort& port,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto shape = ResolveRuntimeShape(port, inputs, outputs);
    if (!shape.ok()) {
        return shape.status();
    }
    return shape->size();
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicDimEqual(
        const ShapeSymbol lhs,
        const ShapeSymbol rhs) noexcept {
    if (lhs == rhs) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    if (lhs.IsStatic() && rhs.IsStatic()) {
        return ShapeConstraintEvaluationResult::kViolated;
    }
    return ShapeConstraintEvaluationResult::kDeferred;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicDimBroadcastable(
        const ShapeSymbol lhs,
        const ShapeSymbol rhs) noexcept {
    if (lhs == rhs) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    if (lhs.IsStatic() && lhs.GetStaticValue() == 1) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    if (rhs.IsStatic() && rhs.GetStaticValue() == 1) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    if (lhs.IsStatic() && rhs.IsStatic()) {
        return ShapeConstraintEvaluationResult::kViolated;
    }
    return ShapeConstraintEvaluationResult::kDeferred;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicDimPositive(
        const ShapeSymbol dim) noexcept {
    if (!dim.IsStatic()) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return dim.GetStaticValue() > 0 ? ShapeConstraintEvaluationResult::kSatisfied
                                    : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicDimEqualConstraint(
        const DimEqualConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto lhs = ResolveSymbolicDim(constraint.lhs, inputs, outputs);
    const auto rhs = ResolveSymbolicDim(constraint.rhs, inputs, outputs);

    if (!lhs || !rhs) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return EvaluateSymbolicDimEqual(*lhs, *rhs);
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicBroadcastableConstraint(
        const DimBroadcastableConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto lhs = ResolveSymbolicDim(constraint.lhs, inputs, outputs);
    const auto rhs = ResolveSymbolicDim(constraint.rhs, inputs, outputs);

    if (!lhs || !rhs) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return EvaluateSymbolicDimBroadcastable(*lhs, *rhs);
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicDimPositiveConstraint(
        const DimPositiveConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto dim = ResolveSymbolicDim(constraint.dim, inputs, outputs);
    if (!dim) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return EvaluateSymbolicDimPositive(*dim);
}

AM_NODISCARD bool HaveIdenticalSymbolicDims(const std::span<const DimLocator> lhs_dims,
                                            const std::span<const DimLocator> rhs_dims,
                                            const std::span<const SymbolicShape> inputs,
                                            const std::span<const SymbolicShape> outputs) {
    if (lhs_dims.size() != rhs_dims.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs_dims.size(); ++i) {
        const auto lhs = ResolveSymbolicDim(lhs_dims[i], inputs, outputs);
        const auto rhs = ResolveSymbolicDim(rhs_dims[i], inputs, outputs);

        if (!lhs || !rhs || *lhs != *rhs) {
            return false;
        }
    }
    return true;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicVolumeConstraint(
        const VolumeEqualConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    if (HaveIdenticalSymbolicDims(constraint.lhs_dims, constraint.rhs_dims, inputs, outputs)) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    // Compute volume from static dims. If any dim is symbolic or overflow
    // occurs, defer to runtime evaluation.
    auto compute_volume = [&](const std::span<const DimLocator> locators,
                              uint64_t& product,
                              bool& deferred) {
        for (const auto& loc: locators) {
            const auto dim = ResolveSymbolicDim(loc, inputs, outputs);

            if (!dim || !dim->IsStatic()) {
                deferred = true;
                return;
            }

            if (CheckOverflowMul(product, static_cast<uint64_t>(dim->GetStaticValue()),
                                 &product)) {
                deferred = true;
                return;
            }
        }
    };

    uint64_t lhs_product = 1;
    uint64_t rhs_product = 1;
    bool lhs_deferred = false;
    bool rhs_deferred = false;
    compute_volume(constraint.lhs_dims, lhs_product, lhs_deferred);
    compute_volume(constraint.rhs_dims, rhs_product, rhs_deferred);

    if (lhs_deferred || rhs_deferred) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }

    return lhs_product == rhs_product ? ShapeConstraintEvaluationResult::kSatisfied
                                      : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicRankEqualConstraint(
        const RankEqualConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto rank = ResolveSymbolicRank(constraint.port, inputs, outputs);

    if (!rank) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return *rank == constraint.target_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                           : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateSymbolicRankAtLeastConstraint(
        const RankAtLeastConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto rank = ResolveSymbolicRank(constraint.port, inputs, outputs);

    if (!rank) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return *rank >= constraint.min_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                        : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeDimEqual(
        const int64_t lhs,
        const int64_t rhs) noexcept {
    return lhs == rhs ? ShapeConstraintEvaluationResult::kSatisfied
                      : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeBroadcastable(
        const int64_t lhs,
        const int64_t rhs) noexcept {
    return lhs == rhs || lhs == 1 || rhs == 1 ? ShapeConstraintEvaluationResult::kSatisfied
                                              : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeDimPositive(
        const int64_t dim) noexcept {
    return dim > 0 ? ShapeConstraintEvaluationResult::kSatisfied
                   : ShapeConstraintEvaluationResult::kViolated;
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeDimEqualConstraint(
        const DimEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto lhs = ResolveRuntimeDim(constraint.lhs, inputs, outputs);
    if (!lhs.ok()) {
        return lhs.status();
    }

    const auto rhs = ResolveRuntimeDim(constraint.rhs, inputs, outputs);
    if (!rhs.ok()) {
        return rhs.status();
    }
    return EvaluateRuntimeDimEqual(*lhs, *rhs);
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeBroadcastableConstraint(
        const DimBroadcastableConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto lhs = ResolveRuntimeDim(constraint.lhs, inputs, outputs);
    if (!lhs.ok()) {
        return lhs.status();
    }

    const auto rhs = ResolveRuntimeDim(constraint.rhs, inputs, outputs);
    if (!rhs.ok()) {
        return rhs.status();
    }
    return EvaluateRuntimeBroadcastable(*lhs, *rhs);
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeDimPositiveConstraint(
        const DimPositiveConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto dim = ResolveRuntimeDim(constraint.dim, inputs, outputs);
    if (!dim.ok()) {
        return dim.status();
    }
    return EvaluateRuntimeDimPositive(*dim);
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeVolumeConstraint(
        const VolumeEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto compute_volume = [&](const std::span<const DimLocator> dims) -> StatusOr<uint64_t> {
        uint64_t product = 1;
        for (const DimLocator& locator: dims) {
            const auto dim = ResolveRuntimeDim(locator, inputs, outputs);
            if (!dim.ok()) {
                return dim.status();
            }
            // Negative dims are malformed runtime bindings; oversized products
            // surface as Overflow instead of aborting the runtime hot path.
            if (*dim < 0) {
                return Status::InvalidArgument(
                        "Shape constraint volume received a negative dimension");
            }
            if (CheckOverflowMul(product, static_cast<uint64_t>(*dim), &product)) {
                return Status::Overflow("Shape constraint volume overflows uint64_t");
            }
        }
        return product;
    };

    const auto lhs = compute_volume(constraint.lhs_dims);
    if (!lhs.ok()) {
        return lhs.status();
    }
    const auto rhs = compute_volume(constraint.rhs_dims);
    if (!rhs.ok()) {
        return rhs.status();
    }
    return *lhs == *rhs ? ShapeConstraintEvaluationResult::kSatisfied
                        : ShapeConstraintEvaluationResult::kViolated;
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeRankEqualConstraint(
        const RankEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto rank = ResolveRuntimeRank(constraint.port, inputs, outputs);
    if (!rank.ok()) {
        return rank.status();
    }
    return *rank == constraint.target_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                           : ShapeConstraintEvaluationResult::kViolated;
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeRankAtLeastConstraint(
        const RankAtLeastConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const auto rank = ResolveRuntimeRank(constraint.port, inputs, outputs);
    if (!rank.ok()) {
        return rank.status();
    }
    return *rank >= constraint.min_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                        : ShapeConstraintEvaluationResult::kViolated;
}

}// namespace

ShapeConstraintEvaluationResult EvaluateShapeConstraint(
        const ShapeConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    auto visitor = overloaded{
            [&](const DimEqualConstraint& dim_equal) {
                return EvaluateSymbolicDimEqualConstraint(dim_equal, inputs, outputs);
            },
            [&](const DimBroadcastableConstraint& broadcastable) {
                return EvaluateSymbolicBroadcastableConstraint(broadcastable, inputs, outputs);
            },
            [&](const VolumeEqualConstraint& volume_equal) {
                return EvaluateSymbolicVolumeConstraint(volume_equal, inputs, outputs);
            },
            [&](const RankEqualConstraint& rank_equal) {
                return EvaluateSymbolicRankEqualConstraint(rank_equal, inputs, outputs);
            },
            [&](const RankAtLeastConstraint& rank_at_least) {
                return EvaluateSymbolicRankAtLeastConstraint(rank_at_least, inputs, outputs);
            },
            [&](const DimPositiveConstraint& dim_positive) {
                return EvaluateSymbolicDimPositiveConstraint(dim_positive, inputs, outputs);
            }};

    return std::visit(visitor, constraint.condition);
}

StatusOr<ShapeConstraintEvaluationResult> EvaluateShapeConstraint(
        const ShapeConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    auto visitor = overloaded{
            [&](const DimEqualConstraint& dim_equal) {
                return EvaluateRuntimeDimEqualConstraint(dim_equal, inputs, outputs);
            },
            [&](const DimBroadcastableConstraint& broadcastable) {
                return EvaluateRuntimeBroadcastableConstraint(broadcastable, inputs, outputs);
            },
            [&](const VolumeEqualConstraint& volume_equal) {
                return EvaluateRuntimeVolumeConstraint(volume_equal, inputs, outputs);
            },
            [&](const RankEqualConstraint& rank_equal) {
                return EvaluateRuntimeRankEqualConstraint(rank_equal, inputs, outputs);
            },
            [&](const RankAtLeastConstraint& rank_at_least) {
                return EvaluateRuntimeRankAtLeastConstraint(rank_at_least, inputs, outputs);
            },
            [&](const DimPositiveConstraint& dim_positive) {
                return EvaluateRuntimeDimPositiveConstraint(dim_positive, inputs, outputs);
            }};

    return std::visit(visitor, constraint.condition);
}

Status ValidateShapeConstraints(const std::span<const ShapeConstraint> constraints,
                                const std::span<const TensorView> inputs,
                                const std::span<const MutableTensorView> outputs) {
    for (const auto& constraint: constraints) {
        const auto result =
                EvaluateShapeConstraint(constraint, inputs, outputs);
        if (!result.ok()) {
            return result.status();
        }

        if (*result == ShapeConstraintEvaluationResult::kViolated) {
            return Status::InvalidArgument(constraint.error_context.empty()
                                                   ? "Runtime shape constraint violated"
                                                   : constraint.error_context);
        }

        if (*result == ShapeConstraintEvaluationResult::kDeferred) {
            return Status::Internal("Runtime shape constraint evaluation returned kDeferred; "
                                    "all dimensions are concrete at runtime, this indicates a bug");
        }
    }
    return Status::Ok();
}

}// namespace aethermind
