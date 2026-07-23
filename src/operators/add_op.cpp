#include "aethermind/operators/add_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/operator_semantics.h"
#include "aethermind/shape_inference/broadcast.h"

#include <span>
#include <string>

namespace aethermind {
Status AddOp::ValidateParams() const {
    return ValidateOperatorParams(Type(), params_);
}

Status AddOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    return InferOperator(Type(), params_, inputs).status();
}

StatusOr<InferenceResult> AddOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    return InferOperator(Type(), params_, inputs);
}

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

}// namespace aethermind