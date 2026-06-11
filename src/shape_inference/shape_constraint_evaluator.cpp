#include "aethermind/shape_inference/shape_constraint_evaluator.h"
#include "aethermind/utils/overflow_check.h"
#include "utils/variant_utils.h"

#include <optional>
#include <string>

namespace aethermind {
namespace {

AM_NODISCARD Status InvalidPortIndex(const TensorPort& port) {
    return Status::InvalidArgument(std::string("Shape constraint references missing ") +
                                   (port.direction == TensorPortType::kInput ? "input" : "output") +
                                   " tensor " + std::to_string(port.tensor_idx));
}

AM_NODISCARD Status InvalidDimIndex(const DimLocator& locator) {
    return Status::InvalidArgument("Shape constraint references missing dimension " +
                                   std::to_string(locator.dim_index));
}

AM_NODISCARD StatusOr<const SymbolicShape*> ResolveShape(const TensorPort& port,
                                                         const std::span<const SymbolicShape> inputs,
                                                         const std::span<const SymbolicShape> outputs) {
    if (port.direction == TensorPortType::kInput) {
        if (port.tensor_idx >= inputs.size()) {
            return InvalidPortIndex(port);
        }
        return &inputs[port.tensor_idx];
    }

    if (port.tensor_idx >= outputs.size()) {
        return InvalidPortIndex(port);
    }
    return &outputs[port.tensor_idx];
}

AM_NODISCARD StatusOr<const TensorView*> ResolveTensor(const TensorPort& port,
                                                       const std::span<const TensorView> inputs) {
    if (port.direction != TensorPortType::kInput || port.tensor_idx >= inputs.size()) {
        return InvalidPortIndex(port);
    }
    return &inputs[port.tensor_idx];
}

AM_NODISCARD StatusOr<const MutableTensorView*> ResolveMutableTensor(
        const TensorPort& port,
        const std::span<const MutableTensorView> outputs) {
    if (port.direction != TensorPortType::kOutput || port.tensor_idx >= outputs.size()) {
        return InvalidPortIndex(port);
    }
    return &outputs[port.tensor_idx];
}

AM_NODISCARD StatusOr<std::optional<ShapeSymbol>> ResolveSymbolicDim(
        const DimLocator& locator,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto shape = ResolveShape(locator.tensor_port, inputs, outputs);
    if (!shape.ok()) {
        return shape.status();
    }

    const auto rank = (*shape)->rank();
    if (!rank.has_value()) {
        return std::optional<ShapeSymbol>{};
    }

    if (locator.dim_index >= *rank) {
        return InvalidDimIndex(locator);
    }
    return std::optional<ShapeSymbol>{(**shape)[locator.dim_index]};
}

AM_NODISCARD StatusOr<int64_t> ResolveRuntimeDim(
        const DimLocator& locator,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    if (locator.tensor_port.direction == TensorPortType::kInput) {
        const auto tensor = ResolveTensor(locator.tensor_port, inputs);
        if (!tensor.ok()) {
            return tensor.status();
        }

        if (locator.dim_index >= (*tensor)->shape().size()) {
            return InvalidDimIndex(locator);
        }
        return (*tensor)->shape()[locator.dim_index];
    }

    const auto tensor = ResolveMutableTensor(locator.tensor_port, outputs);
    if (!tensor.ok()) {
        return tensor.status();
    }

    if (locator.dim_index >= (*tensor)->shape().size()) {
        return InvalidDimIndex(locator);
    }
    return (*tensor)->shape()[locator.dim_index];
}

AM_NODISCARD StatusOr<std::optional<size_t>> ResolveSymbolicRank(
        const TensorPort& port,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto shape = ResolveShape(port, inputs, outputs);
    if (!shape.ok()) {
        return shape.status();
    }
    return (*shape)->rank();
}

AM_NODISCARD StatusOr<size_t> ResolveRuntimeRank(
        const TensorPort& port,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    if (port.direction == TensorPortType::kInput) {
        const auto tensor = ResolveTensor(port, inputs);
        if (!tensor.ok()) {
            return tensor.status();
        }
        return (*tensor)->shape().size();
    }

    const auto tensor = ResolveMutableTensor(port, outputs);
    if (!tensor.ok()) {
        return tensor.status();
    }
    return (*tensor)->shape().size();
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

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateSymbolicDimEqualConstraint(
        const DimEqualConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto lhs = ResolveSymbolicDim(constraint.lhs, inputs, outputs);
    if (!lhs.ok()) {
        return lhs.status();
    }

    const auto rhs = ResolveSymbolicDim(constraint.rhs, inputs, outputs);
    if (!rhs.ok()) {
        return rhs.status();
    }

    if (!*lhs || !*rhs) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return EvaluateSymbolicDimEqual(**lhs, **rhs);
}

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateSymbolicBroadcastableConstraint(
        const DimBroadcastableConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto lhs = ResolveSymbolicDim(constraint.lhs, inputs, outputs);
    if (!lhs.ok()) {
        return lhs.status();
    }

    const auto rhs = ResolveSymbolicDim(constraint.rhs, inputs, outputs);
    if (!rhs.ok()) {
        return rhs.status();
    }

    if (!*lhs || !*rhs) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return EvaluateSymbolicDimBroadcastable(**lhs, **rhs);
}

AM_NODISCARD StatusOr<bool> HaveIdenticalSymbolicDims(const std::span<const DimLocator> lhs_dims,
                                                      const std::span<const DimLocator> rhs_dims,
                                                      const std::span<const SymbolicShape> inputs,
                                                      const std::span<const SymbolicShape> outputs) {
    if (lhs_dims.size() != rhs_dims.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs_dims.size(); ++i) {
        const auto lhs = ResolveSymbolicDim(lhs_dims[i], inputs, outputs);
        if (!lhs.ok()) {
            return lhs.status();
        }

        const auto rhs = ResolveSymbolicDim(rhs_dims[i], inputs, outputs);
        if (!rhs.ok()) {
            return rhs.status();
        }

        if (!*lhs || !*rhs || **lhs != **rhs) {
            return false;
        }
    }
    return true;
}

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateSymbolicVolumeConstraint(
        const VolumeEqualConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto identical_dims = HaveIdenticalSymbolicDims(
            constraint.lhs_dims, constraint.rhs_dims, inputs, outputs);
    if (!identical_dims.ok()) {
        return identical_dims.status();
    }

    if (*identical_dims) {
        return ShapeConstraintEvaluationResult::kSatisfied;
    }

    auto compute_volume = [&](const std::span<const DimLocator> dims,
                              uint64_t& product,
                              bool& deferred) -> Status {
        for (const DimLocator& locator: dims) {
            const auto dim = ResolveSymbolicDim(locator, inputs, outputs);
            if (!dim.ok()) {
                return dim.status();
            }

            if (!*dim || !(*dim)->IsStatic()) {
                deferred = true;
                break;
            }

            if (CheckOverflowMul(product, static_cast<uint64_t>((*dim)->GetStaticValue()), &product)) {
                return Status::InvalidArgument("Shape constraint volume overflows uint64_t");
            }
        }
        return Status::Ok();
    };

    uint64_t lhs_product = 1;
    uint64_t rhs_product = 1;
    bool lhs_deferred = false;
    bool rhs_deferred = false;
    AM_RETURN_IF_ERROR(compute_volume(constraint.lhs_dims, lhs_product, lhs_deferred));
    AM_RETURN_IF_ERROR(compute_volume(constraint.rhs_dims, rhs_product, rhs_deferred));

    if (lhs_deferred || rhs_deferred) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }

    return lhs_product == rhs_product ? ShapeConstraintEvaluationResult::kSatisfied
                                      : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateSymbolicRankEqualConstraint(
        const RankEqualConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto rank = ResolveSymbolicRank(constraint.port, inputs, outputs);
    if (!rank.ok()) {
        return rank.status();
    }

    if (!*rank) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return **rank == constraint.target_rank ? ShapeConstraintEvaluationResult::kSatisfied
                                            : ShapeConstraintEvaluationResult::kViolated;
}

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateSymbolicRankAtLeastConstraint(
        const RankAtLeastConstraint& constraint,
        const std::span<const SymbolicShape> inputs,
        const std::span<const SymbolicShape> outputs) {
    const auto rank = ResolveSymbolicRank(constraint.port, inputs, outputs);
    if (!rank.ok()) {
        return rank.status();
    }

    if (!*rank) {
        return ShapeConstraintEvaluationResult::kDeferred;
    }
    return **rank >= constraint.min_rank ? ShapeConstraintEvaluationResult::kSatisfied
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

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeDimEqualConstraint(
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

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeBroadcastableConstraint(
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

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeVolumeConstraint(
        const VolumeEqualConstraint& constraint,
        const std::span<const TensorView> inputs,
        const std::span<const MutableTensorView> outputs) {
    auto compute_volume = [&](const std::span<const DimLocator> dims) -> StatusOr<uint64_t> {
        uint64_t product = 1;
        for (const DimLocator& locator: dims) {
            const auto dim = ResolveRuntimeDim(locator, inputs, outputs);
            if (!dim.ok()) {
                return dim.status();
            }

            if (CheckOverflowMul(product, static_cast<uint64_t>(*dim), &product)) {
                return Status::InvalidArgument("Shape constraint volume overflows uint64_t");
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

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeRankEqualConstraint(
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

AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateRuntimeRankAtLeastConstraint(
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

StatusOr<ShapeConstraintEvaluationResult> EvaluateShapeConstraint(const ShapeConstraint& constraint,
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

StatusOr<ShapeConstraintEvaluationResult> EvaluateShapeConstraint(const ShapeConstraint& constraint,
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
        if (!result.ok()) {
            return result.status();
        }

        if (*result == ShapeConstraintEvaluationResult::kViolated) {
            return Status::InvalidArgument(constraint.error_context.empty() ? "Runtime shape constraint violated"
                                                                            : constraint.error_context);
        }
    }
    return Status::Ok();
}

}// namespace aethermind
