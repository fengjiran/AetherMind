#include "aethermind/operators/silu_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"

#include <span>
#include <string>

namespace aethermind {

Status SiluOp::ValidateParams() const {
    return Status::Ok();
}

Status SiluOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 1) {
        return Status::InvalidArgument(
                "Silu expects exactly 1 input, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    if (input_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Silu only supports float32 input in Phase 1");
    }

    return Status::Ok();
}

StatusOr<InferenceResult> SiluOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 1) {
        return Status::InvalidArgument(
                "Silu expects exactly 1 shape input, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    if (input_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Silu only supports float32 input in Phase 1");
    }

    // SiLU is element-wise: output shape == input shape. No broadcasting,
    // no deferred axes, no runtime shape checks.
    TensorSpec output_spec{
            .dtype = DataType::Float32(),
            .shape = input_spec.shape,
    };

    return InferenceResult{
            .outputs = {std::move(output_spec)},
    };
}

Status SiluOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("Silu Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kSilu,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("Silu Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status SiluOp::Run(KernelContext& ctx,
                   const RuntimeBindingContext& bindings,
                   size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("Silu Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 1) {
        return Status::InvalidArgument(
                "Silu requires 1 input tensor binding, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "Silu requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    // Kernel not yet implemented. Binding contracts are validated above so
    // that the semantic layer is exercised; execution will be enabled once a
    // CPU kernel is registered and the kernel-params contract is defined.
    return Status::Unimplemented("Silu kernel not yet implemented");
}

AM_REGISTER_OPERATOR(OpType::kSilu, SiluOp)

}// namespace aethermind
