#include "aethermind/operators/add_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

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

// Shape inference and constraint analysis for the Add operator.
//
// Validation order (parameter-before-input precedence):
//   1. Validate OpParams variant type (must be AddParams).
//   2. Validate input count via ValidateInferenceInputCount (exactly 2:
//      lhs and rhs, checked against the operator schema).
//   3. Validate dtype homogeneity: lhs and rhs must share the same dtype.
//   4. Validate dtype support: the shared dtype must belong to the
//      Add-supported set (float32, float64, bfloat16, int32, int64).
//
// Inference algorithm:
//   - Compute the output shape via InferBroadcastShape (NumPy-style
//     right-aligned broadcasting over lhs and rhs symbolic shapes).
//   - Output dtype equals the (validated-equal) input dtype.
//   - For each broadcast axis pair that cannot be proven compatible
//     statically, emit a deferred DimBroadcastableConstraint verified
//     at runtime by the Executor.
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
