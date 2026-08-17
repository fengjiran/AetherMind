#include "aethermind/operators/ops/elementwise_mul_op.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/broadcast.h"

namespace aethermind {

namespace detail {

StatusOr<InferenceResult> InferElementwiseMul(const OpParams& params,
                                              std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<ElementwiseMulParams>(params)) {
        return Status::InvalidArgument(
                "ElementwiseMul node requires ElementwiseMulParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kElementwiseMul, inputs));

    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (!IsElementwiseMulSupportedDType(lhs_spec.dtype)) {
        return Status::InvalidArgument(
                MakeElementwiseMulUnsupportedDTypeMessage("ElementwiseMul lhs"));
    }

    if (!IsElementwiseMulSupportedDType(rhs_spec.dtype)) {
        return Status::InvalidArgument(
                MakeElementwiseMulUnsupportedDTypeMessage("ElementwiseMul rhs"));
    }

    auto broadcast_result = InferBroadcastShape(
            lhs_spec.shape, rhs_spec.shape);
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
                "ElementwiseMul dimensions must be broadcastable");
    }
    return result;
}

}// namespace detail

}// namespace aethermind
