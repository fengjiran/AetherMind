#include "aethermind/operators/silu_mul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

namespace aethermind {

Status SiluMulOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("SiluMul Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kSiluMul,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("SiluMul Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status SiluMulOp::Run(KernelContext& ctx,
                      const RuntimeBindingContext& bindings,
                      size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("SiluMul Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "SiluMul requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "SiluMul requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    // Kernel not yet implemented. Binding contracts are validated above so
    // that the semantic layer is exercised; execution will be enabled once a
    // CPU kernel is registered and the kernel-params contract is defined.
    return Status::Unimplemented("SiluMul kernel not yet implemented");
}

AM_REGISTER_OPERATOR(OpType::kSiluMul, SiluMulOp)


namespace detail {

StatusOr<InferenceResult> InferSiluMul(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<SiluMulParams>(params)) {
        return Status::InvalidArgument("SiluMul node requires SiluMulParams");
    }
    if (inputs.size() != 2) {
        return Status::InvalidArgument("SiluMul requires exactly 2 inputs");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != DataType::Float32() && lhs_spec.dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument("SiluMul lhs must be float32 or bfloat16");
    }
    if (rhs_spec.dtype != DataType::Float32() && rhs_spec.dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument("SiluMul rhs must be float32 or bfloat16");
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
                 "SiluMul dimensions must be broadcastable"});
    }
    return result;
}

}// namespace detail

}// namespace aethermind
