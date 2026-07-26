#include <optional>

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_constraint_evaluator.h"
#include "aethermind/utils/overflow_check.h"
#include "utils/variant_utils.h"

namespace aethermind::detail {
namespace {

// Builds the VolumeEqualConstraint covering every input dim (lhs) and every
// output dim (rhs). An empty list represents scalar volume 1.
VolumeEqualConstraint MakeVolumeEqualConstraint(const SymbolicShape& input_shape,
                                                const SymbolicShape& output_shape) {
    VolumeEqualConstraint constraint;
    constraint.lhs_dims.reserve(*input_shape.rank());
    for (size_t i = 0; i < *input_shape.rank(); ++i) {
        DimLocator locator;
        locator.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 0};
        locator.dim_index = i;
        constraint.lhs_dims.push_back(locator);
    }

    constraint.rhs_dims.reserve(*output_shape.rank());
    for (size_t i = 0; i < *output_shape.rank(); ++i) {
        DimLocator locator;
        locator.tensor_port = {.direction = TensorPortType::kOutput, .tensor_idx = 0};
        locator.dim_index = i;
        constraint.rhs_dims.push_back(locator);
    }
    return constraint;
}

// Multiplies a range of static dims into *product using checked uint64_t
// arithmetic. Returns Status::Overflow if the product overflows. Caller is
// responsible for ensuring all dims are static before invoking.
Status MultiplyStaticDims(const SymbolicShape& shape, uint64_t* product) {
    for (size_t i = 0; i < *shape.rank(); ++i) {
        const ShapeSymbol& dim = shape[i];
        if (const auto value = static_cast<uint64_t>(dim.GetStaticValue());
            CheckOverflowMul(*product, value, product)) {
            return Status::Overflow("Reshape volume product overflows");
        }
    }
    return Status::Ok();
}

// Returns true if every dim in shape is static, excluding skip_index.
bool AllDimsStaticExcept(const SymbolicShape& shape, size_t skip_index) {
    for (size_t i = 0; i < *shape.rank(); ++i) {
        if (i == skip_index) {
            continue;
        }
        if (!shape[i].IsStatic()) {
            return false;
        }
    }
    return true;
}

// Attempts to statically resolve the infer dimension in output_shape.
// On success, replaces output_shape[infer_index] with the computed quotient.
// If not all required dims are static, leaves Unknown() intact and returns Ok.
Status TryResolveInferDim(const SymbolicShape& input_shape,
                          SymbolicShape& output_shape,
                          size_t infer_index) {
    if (!input_shape.IsStatic() || !AllDimsStaticExcept(output_shape, infer_index)) {
        return Status::Ok();
    }

    uint64_t input_volume = 1;
    AM_RETURN_IF_ERROR(MultiplyStaticDims(input_shape, &input_volume));

    uint64_t non_infer_product = 1;
    for (size_t i = 0; i < *output_shape.rank(); ++i) {
        if (i == infer_index) {
            continue;
        }
        if (const auto value = static_cast<uint64_t>(output_shape[i].GetStaticValue());
            CheckOverflowMul(non_infer_product, value, &non_infer_product)) {
            return Status::Overflow("Reshape non-infer product overflows");
        }
    }

    if (non_infer_product == 0) {
        // Zero-volume ambiguity: input volume zero is ambiguous and
        // non-zero is impossible. Reject.
        return Status::InvalidArgument(
                "Reshape infer marker is ambiguous for zero-volume shapes");
    }

    if (input_volume % non_infer_product != 0) {
        return Status::InvalidArgument(
                "Reshape volume not divisible by non-infer product");
    }

    const uint64_t quotient = input_volume / non_infer_product;
    if (quotient > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return Status::Overflow("Reshape infer quotient overflows int64_t");
    }
    // Replace Unknown at infer_index with the static quotient. Zero is
    // a valid static quotient.
    output_shape[infer_index] = ShapeSymbol::CreateFromValue(static_cast<int64_t>(quotient));
    return Status::Ok();
}

}// namespace

// Reshape semantic inference.
//
// Validation order (per plan, parameter-before-input precedence):
//   1. require ReshapeParams variant
//   2. parameter-only invariants: literal values non-negative, at most one
//      infer marker
//   3. require exactly one input
//   4. require ranked input
//   5. validate every ReshapeInputDim.axis against input rank
//
// Inference algorithm:
//   - Build output dims in target order: literal -> CreateFromValue;
//     input reference -> copy exact input ShapeSymbol (identity preserved);
//     infer -> Unknown() unless statically resolvable.
//   - For one infer marker, statically resolve when every input dim and every
//     non-infer output dim is static. Compute products with checked uint64_t
//     arithmetic; return Status::Overflow on overflow. Non-infer product == 0
//     is rejected (zero-volume ambiguity). Otherwise, require exact
//     divisibility, require quotient to fit int64_t, emit static quotient.
//   - If static resolution impossible, emit ShapeSymbol::Unknown().
//   - Construct one VolumeEqualConstraint covering every input dim and every
//     output dim. Evaluate symbolically: kViolated -> reject;
//     kSatisfied -> omit; kDeferred -> persist exactly one deferred check.
StatusOr<InferenceResult> InferReshape(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* reshape_params = std::get_if<ReshapeParams>(&params);
    if (reshape_params == nullptr) {
        return Status::InvalidArgument("Reshape node requires ReshapeParams");
    }

    // Parameter-only invariants: validate before input checks so a malformed
    // ReshapeParams is reported even when the input arity is also wrong.
    size_t infer_count = 0;
    for (const auto& dim: reshape_params->target_shape) {
        auto visitor = overloaded{
                [&](const ReshapeLiteralDim& d) {
                    if (d.value < 0) {
                        return Status::InvalidArgument(
                                "Reshape literal dim must be non-negative");
                    }
                    return Status::Ok();
                },
                [&](const ReshapeInputDim&) {
                    return Status::Ok();
                },
                [&](const ReshapeInferDim) {
                    ++infer_count;
                    return Status::Ok();
                },
        };
        AM_RETURN_IF_ERROR(std::visit(visitor, dim));
    }

    if (infer_count > 1) {
        return Status::InvalidArgument(
                "Reshape target_shape may contain at most one infer marker");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kReshape, inputs));

    const TensorSpec& input = inputs[0];
    if (!input.shape.IsRanked()) {
        return Status::InvalidArgument("Reshape input must be ranked");
    }
    const size_t input_rank = *input.shape.rank();

    // Validate input-axis references against input rank.
    for (const auto& dim: reshape_params->target_shape) {
        if (const auto* ref = std::get_if<ReshapeInputDim>(&dim);
            ref != nullptr && static_cast<size_t>(ref->axis) >= input_rank) {
            return Status::InvalidArgument(
                    "Reshape input-axis reference out of range");
        }
    }

    // Build output dims in target order. Static input dims and all non-infer
    // output dims must be known for static infer resolution; we record them
    // alongside the output construction.
    std::vector<ShapeSymbol> output_dims;
    output_dims.reserve(reshape_params->target_shape.size());
    std::optional<size_t> infer_index;
    for (size_t i = 0; i < reshape_params->target_shape.size(); ++i) {
        const auto& dim = reshape_params->target_shape[i];
        auto visitor = overloaded{
                [&](const ReshapeLiteralDim& d) {
                    output_dims.push_back(ShapeSymbol::CreateFromValue(d.value));
                },
                [&](const ReshapeInputDim& d) {
                    // Copy exact input ShapeSymbol to preserve
                    // symbolic identity (not just static value).
                    output_dims.push_back(input.shape[d.axis]);
                },
                [&](const ReshapeInferDim) {
                    infer_index = i;
                    output_dims.push_back(ShapeSymbol::Unknown());
                },
        };
        std::visit(visitor, dim);
    }
    SymbolicShape output_shape(std::move(output_dims));

    // Static infer resolution (only when exactly one infer marker present).
    if (infer_index.has_value()) {
        AM_RETURN_IF_ERROR(TryResolveInferDim(input.shape, output_shape, *infer_index));
    }

    // Construct one VolumeEqualConstraint covering every input dim (lhs) and
    // every output dim (rhs). Empty lists represent scalar volume 1.
    ShapeConstraint constraint{
            .condition = MakeVolumeEqualConstraint(input.shape, output_shape),
            .error_context = "Reshape volume mismatch",
    };

    // Evaluate the constraint symbolically against the current shapes.
    const std::array<SymbolicShape, 1> input_shapes{input.shape};
    const std::array<SymbolicShape, 1> output_shapes{output_shape};
    const ShapeConstraintEvaluationResult eval_result = EvaluateShapeConstraint(
            constraint, input_shapes, output_shapes);

    InferenceResult result;
    result.outputs.push_back({input.dtype, output_shape});

    switch (eval_result) {
        case ShapeConstraintEvaluationResult::kViolated:
            return Status::InvalidArgument("Reshape input/output volumes are not equal");
        case ShapeConstraintEvaluationResult::kSatisfied:
            // Static proof succeeded; no deferred runtime check needed.
            break;
        case ShapeConstraintEvaluationResult::kDeferred:
            // When every input and output dim is static, deferred can only
            // result from overflow during the symbolic volume product; surface
            // it eagerly as Status::Overflow instead of persisting a deferred
            // check that would similarly fail at runtime.
            if (input.shape.IsStatic() && output_shape.IsStatic()) {
                return Status::Overflow("Reshape volume product overflows");
            }
            // Persist exactly one deferred runtime check for runtime
            // enforcement.
            result.runtime_checks.push_back(std::move(constraint));
            break;
    }
    return result;
}

}// namespace aethermind::detail
