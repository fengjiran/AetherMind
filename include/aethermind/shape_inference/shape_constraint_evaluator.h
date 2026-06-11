#ifndef AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H
#define AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"

#include <span>

namespace aethermind {

/// Evaluates a shape constraint against symbolic tensor shapes.
///
/// Returns kDeferred when the current symbolic facts cannot prove or disprove
/// the constraint.
AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateShapeConstraint(
        const ShapeConstraint& constraint,
        std::span<const SymbolicShape> inputs,
        std::span<const SymbolicShape> outputs);

/// Evaluates a shape constraint against concrete runtime tensor views.
AM_NODISCARD StatusOr<ShapeConstraintEvaluationResult> EvaluateShapeConstraint(
        const ShapeConstraint& constraint,
        std::span<const TensorView> inputs,
        std::span<const MutableTensorView> outputs);

/// Validates all concrete runtime constraints and converts violations to Status.
AM_NODISCARD Status ValidateShapeConstraints(
        std::span<const ShapeConstraint> constraints,
        std::span<const TensorView> inputs,
        std::span<const MutableTensorView> outputs);

}// namespace aethermind

#endif// AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H
