#include "aethermind/operators/ops/add_op.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/broadcast.h"

namespace aethermind {
namespace detail {

// Validate parameters before input structure so inference failures have stable
// precedence. Symbolic broadcast compatibility is deferred to runtime.
StatusOr<InferenceResult> InferAdd(const OpParams& params,
                                   std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<AddParams>(params)) {
        return Status::InvalidArgument("Add node requires AddParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kAdd, inputs));

    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != rhs_spec.dtype) {
        return Status::InvalidArgument("Add inputs must have the same dtype");
    }

    if (!IsAddSupportedDType(lhs_spec.dtype)) {
        return Status::InvalidArgument(MakeAddUnsupportedDTypeMessage("Add"));
    }

    auto broadcast_result =
            InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    InferenceResult result;
    result.outputs.emplace_back(lhs_spec.dtype, broadcast_result->output_shape);
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.emplace_back(
                DimBroadcastableConstraint{
                        {{TensorPortType::kInput, 0},
                         deferred.lhs_axis},
                        {{TensorPortType::kInput, 1},
                         deferred.rhs_axis}},
                "Add dimensions must be broadcastable");
    }
    return result;
}

}// namespace detail

}// namespace aethermind
