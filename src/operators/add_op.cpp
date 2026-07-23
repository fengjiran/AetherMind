#include "aethermind/operators/add_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include <span>
#include <string>

namespace aethermind {
Status AddOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("Add Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(OpType::kAdd,
                                                         ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("Add Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status AddOp::Run(KernelContext& ctx,
                  const RuntimeBindingContext& bindings,
                  size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("Add Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument("Add requires 2 input tensor bindings, got " +
                                       std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument("Add requires 1 output tensor binding, got " +
                                       std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

// Registers AddOp as the constructor for OpType::kAdd with the operator
// factory so graph builders can instantiate it by type.
AM_REGISTER_OPERATOR(OpType::kAdd, AddOp)


namespace detail {

StatusOr<InferenceResult> InferAdd(const OpParams& params,
                                   std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<AddParams>(params)) {
        return Status::InvalidArgument("Add node requires AddParams");
    }
    if (inputs.size() != 2) {
        return Status::InvalidArgument("Add requires exactly 2 inputs");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != rhs_spec.dtype) {
        return Status::InvalidArgument("Add inputs must have the same dtype");
    }
    if (!IsAddSupportedDType(lhs_spec.dtype)) {
        return Status::InvalidArgument(MakeAddUnsupportedDTypeMessage("Add"));
    }
    auto broadcast_result = InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }
    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, broadcast_result->output_shape});
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimBroadcastableConstraint{{{TensorPortType::kInput, 0}, deferred.lhs_axis},
                                           {{TensorPortType::kInput, 1}, deferred.rhs_axis}},
                "Add dimensions must be broadcastable"});
    }
    return result;
}

}// namespace detail

}// namespace aethermind
