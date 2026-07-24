#include "aethermind/operators/elementwise_mul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include <span>
#include <string>

namespace aethermind {

Status ElementwiseMulOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument(
                "ElementwiseMul Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kElementwiseMul,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("ElementwiseMul Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status ElementwiseMulOp::Run(KernelContext& ctx,
                             const RuntimeBindingContext& bindings,
                             size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("ElementwiseMul Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "ElementwiseMul requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "ElementwiseMul requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kElementwiseMul, ElementwiseMulOp)


namespace detail {

StatusOr<InferenceResult> InferElementwiseMul(const OpParams& params,
                                              std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<ElementwiseMulParams>(params)) {
        return Status::InvalidArgument("ElementwiseMul node requires ElementwiseMulParams");
    }
    if (inputs.size() != 2) {
        return Status::InvalidArgument("ElementwiseMul requires exactly 2 inputs");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != DataType::Float32() && lhs_spec.dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument("ElementwiseMul lhs must be float32 or bfloat16");
    }
    if (rhs_spec.dtype != DataType::Float32() && rhs_spec.dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument("ElementwiseMul rhs must be float32 or bfloat16");
    }
    auto broadcast_result = InferBroadcastShape(
            lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, broadcast_result->output_shape});
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(
                {DimBroadcastableConstraint{
                         {{TensorPortType::kInput, 0},
                          deferred.lhs_axis},
                         {{TensorPortType::kInput, 1},
                          deferred.rhs_axis}},
                 "ElementwiseMul dimensions must be broadcastable"});
    }
    return result;
}

}// namespace detail

}// namespace aethermind
