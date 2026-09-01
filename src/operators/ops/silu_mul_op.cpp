#include "aethermind/operators/ops/silu_mul_op.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/broadcast.h"

namespace aethermind {

namespace detail {

StatusOr<InferenceResult> InferSiluMul(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<SiluMulParams>(params)) {
        return Status::InvalidArgument("SiluMul node requires SiluMulParams");
    }
    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kSiluMul, inputs));

    const TensorSpec& gate = inputs[0];
    const TensorSpec& up = inputs[1];

    // No implicit conversion policy is declared for the fused operation.
    if (gate.dtype != up.dtype) {
        return Status::InvalidArgument(
                "SiLUMul currently requires gate and up to use the same dtype");
    }

    if (!IsSiluMulSupportedDType(gate.dtype)) {
        return Status::InvalidArgument(
                MakeSiluMulUnsupportedDTypeMessage("SiluMul lhs"));
    }

    auto broadcast_result = InferBroadcastShape(
            gate.shape, up.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    InferenceResult result;
    result.outputs.emplace_back(gate.dtype, std::move(broadcast_result->output_shape));
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.emplace_back(
                DimBroadcastableConstraint{
                        {{TensorPortType::kInput, 0},
                         deferred.lhs_axis},
                        {{TensorPortType::kInput, 1},
                         deferred.rhs_axis}},
                "SiluMul dimensions must be broadcastable");
    }
    return result;
}

} // namespace detail

} // namespace aethermind
