#ifndef AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H
#define AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"

#include <span>

namespace aethermind {

/// @brief Evaluates a shape constraint against symbolic tensor shapes.
///
/// Returns kDeferred when the current symbolic facts cannot prove or disprove
/// the constraint.
/// @param constraint The shape constraint to evaluate.
/// @param inputs     Symbolic shapes of operator inputs.
/// @param outputs    Symbolic shapes of operator outputs.
/// @return kSatisfied, kViolated, or kDeferred.
AM_NODISCARD ShapeConstraintEvaluationResult EvaluateShapeConstraint(
        const ShapeConstraint& constraint,
        std::span<const SymbolicShape> inputs,
        std::span<const SymbolicShape> outputs);

/// @brief Evaluates a shape constraint against concrete runtime tensor views.
/// @param constraint The shape constraint to evaluate.
/// @param inputs     Concrete runtime input tensor views.
/// @param outputs    Concrete runtime output tensor views.
/// @return kSatisfied or kViolated (never kDeferred at runtime), or an error
///         Status (kInvalidArgument) when the constraint references a tensor
///         port or dimension outside the bound spans. The runtime hot path
///         never aborts on a malformed constraint; plan-build validation
///         rejects out-of-range references up front.
StatusOr<ShapeConstraintEvaluationResult> EvaluateShapeConstraint(
        const ShapeConstraint& constraint,
        std::span<const TensorView> inputs,
        std::span<const MutableTensorView> outputs);

/// @brief Validates all concrete runtime constraints and converts violations to Status.
/// @param constraints The set of shape constraints to validate.
/// @param inputs      Concrete runtime input tensor views.
/// @param outputs     Concrete runtime output tensor views.
/// @return OkStatus if all constraints are satisfied; kInvalidArgument on
///         violation or on a constraint referencing a port/dimension outside
///         the bound spans; kInternal if evaluation unexpectedly returns
///         kDeferred.
Status ValidateShapeConstraints(
        std::span<const ShapeConstraint> constraints,
        std::span<const TensorView> inputs,
        std::span<const MutableTensorView> outputs);

}// namespace aethermind

#endif// AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H
