#include "aethermind/operators/rmsnorm_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/operator_semantics.h"

#include <span>
#include <string>

namespace aethermind {
Status RmsNormOp::ValidateParams() const {
    return ValidateOperatorParams(Type(), params_);
}

Status RmsNormOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    return AnalyzeOperator(Type(), params_, inputs).status();
}

StatusOr<InferenceResult> RmsNormOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    return AnalyzeOperator(Type(), params_, inputs);
}

Status RmsNormOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("RmsNorm Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kRmsNorm,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("RmsNorm Prepare resolved a kernel with null fn");
    }
    const auto eps_bytes = std::as_bytes(std::span{&params_.eps, size_t{1}});
    resolved_kernel_.attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status RmsNormOp::Run(KernelContext& ctx,
                      const RuntimeBindingContext& bindings,
                      size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("RmsNorm Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "RmsNorm requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)

}// namespace aethermind
