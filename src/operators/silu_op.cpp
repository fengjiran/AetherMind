#include "aethermind/operators/silu_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_registry.h"

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include <span>
#include <string>

namespace aethermind {

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


namespace detail {

StatusOr<InferenceResult> InferSilu(const OpParams& params,
                                    std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<SiluParams>(params)) {
        return Status::InvalidArgument("Silu node requires SiluParams");
    }
    if (inputs.size() != 1) {
        return Status::InvalidArgument("Silu requires exactly 1 input");
    }
    if (inputs[0].dtype != DataType::Float32() && inputs[0].dtype != DataType::BFloat(16)) {
        return Status::InvalidArgument("Silu input must be float32 or bfloat16");
    }
    InferenceResult result;
    result.outputs.push_back(inputs[0]);
    return result;
}

}// namespace detail

}// namespace aethermind
