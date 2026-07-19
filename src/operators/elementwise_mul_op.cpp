#include "aethermind/operators/elementwise_mul_op.h"
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

Status ElementwiseMulOp::ValidateParams() const {
    return ValidateOperatorParams(Type(), params_);
}

Status ElementwiseMulOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    return AnalyzeOperator(Type(), params_, inputs).status();
}

StatusOr<InferenceResult> ElementwiseMulOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    return AnalyzeOperator(Type(), params_, inputs);
}

Status ElementwiseMulOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("ElementwiseMul Prepare requires OperatorContext.backend");
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

}// namespace aethermind
