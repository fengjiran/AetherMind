#include "aethermind/shape_inference/shape_constraint_evaluator.h"
#include "aethermind/utils/overflow_check.h"
#include "utils/logging.h"
#include "utils/variant_utils.h"

#include <optional>

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

AM_NODISCARD const TensorView* ResolveTensor(const TensorPort& port,
                                             const std::span<const TensorView> inputs) {
    AM_CHECK(port.direction == TensorPortType::kInput && port.tensor_idx < inputs.size(),
             "Shape constraint references missing input tensor %zu", port.tensor_idx);
    return &inputs[port.tensor_idx];
}

AM_NODISCARD const MutableTensorView* ResolveMutableTensor(
        const TensorPort& port,
        const std::span<const MutableTensorView> outputs) {
    AM_CHECK(port.direction == TensorPortType::kOutput && port.tensor_idx < outputs.size(),
             "Shape constraint references missing output tensor %zu", port.tensor_idx);
    return &outputs[port.tensor_idx];
}

AM_NODISCARD std::optional<ShapeSymbol> ResolveSymbolicDim(
        const DimLocator& locator,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto* shape = ResolveShape(locator.tensor_port, inputs, outputs);

    const auto rank = shape->rank();
    if (!rank.has_value()) {
        return std::optional<ShapeSymbol>{};
    }

    AM_CHECK(locator.dim_index < *rank,
             "Shape constraint references missing dimension %zu", locator.dim_index);
    return std::optional<ShapeSymbol>{(*shape)[locator.dim_index]};
}

AM_NODISCARD int64_t ResolveRuntimeDim(
        const DimLocator& locator,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    if (locator.tensor_port.direction == TensorPortType::kInput) {
        const auto* tensor = ResolveTensor(locator.tensor_port, inputs);
        AM_CHECK(locator.dim_index < tensor->shape().size(),
                 "Shape constraint references missing dimension %zu", locator.dim_index);
        return tensor->shape()[locator.dim_index];
    }

    const auto* tensor = ResolveMutableTensor(locator.tensor_port, outputs);
    AM_CHECK(locator.dim_index < tensor->shape().size(),
             "Shape constraint references missing dimension %zu", locator.dim_index);
    return tensor->shape()[locator.dim_index];
}

AM_NODISCARD std::optional<size_t> ResolveSymbolicRank(
        const TensorPort& port,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    return ResolveShape(port, inputs, outputs)->rank();
}

AM_NODISCARD size_t ResolveRuntimeRank(
        const TensorPort& port,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    if (port.direction == TensorPortType::kInput) {
        return ResolveTensor(port, inputs)->shape().size();
    }
    return ResolveMutableTensor(port, outputs)->shape().size();
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
    auto compute_volume = [&](const std::span<const DimLocator> dims,
                              uint64_t& product,
                              bool& deferred) {
        for (const DimLocator& locator: dims) {
            const auto dim = ResolveSymbolicDim(locator, inputs, outputs);

            if (!dim || !dim->IsStatic()) {
                deferred = true;
                return;
            }

            if (CheckOverflowMul(product, static_cast<uint64_t>(dim->GetStaticValue()), &product)) {
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

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeDimEqual(const int64_t lhs,
                                                                     const int64_t rhs) noexcept {
    return lhs == rhs ? ShapeConstraintEvaluationResult::kSatisfied
                      : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeBroadcastable(const int64_t lhs,
                                                                          const int64_t rhs) noexcept {
    return lhs == rhs || lhs == 1 || rhs == 1 ? ShapeConstraintEvaluationResult::kSatisfied
                                              : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeDimEqualConstraint(
        const DimEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const int64_t lhs = ResolveRuntimeDim(constraint.lhs, inputs, outputs);
    const int64_t rhs = ResolveRuntimeDim(constraint.rhs, inputs, outputs);
    return EvaluateRuntimeDimEqual(lhs, rhs);
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeBroadcastableConstraint(
        const DimBroadcastableConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const int64_t lhs = ResolveRuntimeDim(constraint.lhs, inputs, outputs);
    const int64_t rhs = ResolveRuntimeDim(constraint.rhs, inputs, outputs);
    return EvaluateRuntimeBroadcastable(lhs, rhs);
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeVolumeConstraint(
        const VolumeEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    auto compute_volume = [&](const std::span<const DimLocator> dims) -> uint64_t {
        uint64_t product = 1;
        for (const DimLocator& locator: dims) {
            const int64_t dim = ResolveRuntimeDim(locator, inputs, outputs);
            AM_CHECK(!CheckOverflowMul(product, static_cast<uint64_t>(dim), &product),
                     "Shape constraint volume overflows uint64_t");
        }
        return product;
    };

    const uint64_t lhs = compute_volume(constraint.lhs_dims);
    const uint64_t rhs = compute_volume(constraint.rhs_dims);
    return lhs == rhs ? ShapeConstraintEvaluationResult::kSatisfied
                      : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeRankEqualConstraint(
        const RankEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const size_t rank = ResolveRuntimeRank(constraint.port, inputs, outputs);
    return rank == constraint.target_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                          : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD ShapeConstraintEvaluationResult EvaluateRuntimeRankAtLeastConstraint(
        const RankAtLeastConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    const size_t rank = ResolveRuntimeRank(constraint.port, inputs, outputs);
    return rank >= constraint.min_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                       : ShapeConstraintEvaluationResult::kViolated;
}

}// namespace

ShapeConstraintEvaluationResult EvaluateShapeConstraint(const ShapeConstraint& constraint,
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
    };
    return std::visit(visitor, constraint.condition);
}

ShapeConstraintEvaluationResult EvaluateShapeConstraint(const ShapeConstraint& constraint,
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
    };
    return std::visit(visitor, constraint.condition);
}

Status ValidateShapeConstraints(const std::span<const ShapeConstraint> constraints,
                                const std::span<const TensorView> inputs,
                                const std::span<const MutableTensorView> outputs) {
    for (const auto& constraint: constraints) {
        const auto result = EvaluateShapeConstraint(constraint, inputs, outputs);
        if (result == ShapeConstraintEvaluationResult::kViolated) {
            return Status::InvalidArgument(constraint.error_context.empty() ? "Runtime shape constraint violated"
                                                                            : constraint.error_context);
        }

        if (result == ShapeConstraintEvaluationResult::kDeferred) {
            return Status::Internal("Runtime shape constraint evaluation returned kDeferred; "
                                    "all dimensions are concrete at runtime, this indicates a bug");
        }
    }
    return Status::Ok();
}

}// namespace aethermind
