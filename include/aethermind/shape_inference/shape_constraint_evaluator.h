#ifndef AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H
#define AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "container/array_view.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>

namespace aethermind {

/// @brief ShapeSymbol value mapped to the concrete runtime dimension value it
///        instantiated to. Shared across ports so symbolic identity drift is
///        caught across tensors too.
using SymbolValueMap = std::unordered_map<int64_t, int64_t>;

/// @brief Validates a concrete runtime dtype/shape against a plan-time spec.
///
/// Shared by external-binding validation (BuildExecutionBindings cold path)
/// and per-step binding premises checks. On static dimensions the concrete
/// value must equal the plan value; on symbolic dimensions the concrete value
/// is recorded into `symbol_values` and a second binding of the same symbol
/// to a different value is rejected. Unknown dims carry no premise.
///
/// @param spec   Plan-time spec the concrete tensor must satisfy.
/// @param dtype  Concrete dtype.
/// @param shape  Concrete shape.
/// @param role   Human-readable label used in error messages (e.g. "input").
/// @param port_index Port index used in error messages.
/// @param symbol_values Shared symbol-to-value map (across ports of one step).
/// @return OkStatus, or kInvalidArgument on the first drift.
Status ValidateConcreteShapeAgainstSpec(const TensorSpec& spec,
                                        DataType dtype,
                                        IntArrayView shape,
                                        std::string_view role,
                                        size_t port_index,
                                        SymbolValueMap& symbol_values);

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

/// @brief Validates that runtime tensor bindings preserve the plan-time
///        TensorSpec premises that statically-proven shape constraints rely on.
///
/// Constraints proven satisfied at plan time (kSatisfied) are pruned from
/// runtime_checks, so a plan bound to incompatible tensors would silently
/// break them. This closes that gap by checking, per bound view, that:
/// - dtype matches the plan spec;
/// - rank matches the plan spec when the spec's rank is statically known;
/// - static dimensions match their plan values;
/// - dimensions sharing one ShapeSymbol (symbolic identity) still agree at
///   runtime, across ports of the same step.
///
/// Invalid views (default-constructed test stubs) are skipped: they carry no
/// shape premises to compare against.
///
/// @param input_specs  Compact input specs from the plan step.
/// @param output_specs Output specs from the plan step.
/// @param inputs       Concrete runtime input tensor views.
/// @param outputs      Concrete runtime output tensor views.
/// @return OkStatus when all premises hold; kInvalidArgument on the first
///         drift, with a message identifying the port and dimension.
Status ValidateTensorBindingPremises(
        std::span<const TensorSpec> input_specs,
        std::span<const TensorSpec> output_specs,
        std::span<const TensorView> inputs,
        std::span<const MutableTensorView> outputs);

} // namespace aethermind

#endif // AETHERMIND_SHAPE_INFERENCE_SHAPE_CONSTRAINT_EVALUATOR_H
