#include "aethermind/operators/silu_mul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include <span>
#include <string>

namespace aethermind {

Status SiluMulOp::ValidateParams() const {
    return Status::Ok();
}

Status SiluMulOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "SiluMul expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& gate_spec = inputs[0];
    const auto& up_spec = inputs[1];

    if (gate_spec.dtype != DataType::Float32() || up_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("SiluMul only supports float32 inputs in Phase 1");
    }

    auto broadcast_result = InferBroadcastShape(gate_spec.shape, up_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    return Status::Ok();
}

StatusOr<InferenceResult> SiluMulOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "SiluMul expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& gate_spec = inputs[0];
    const auto& up_spec = inputs[1];

    if (gate_spec.dtype != DataType::Float32() || up_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("SiluMul only supports float32 inputs in Phase 1");
    }

    auto broadcast_result = InferBroadcastShape(gate_spec.shape, up_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    TensorSpec output_spec{
            .dtype = DataType::Float32(),
            .shape = broadcast_result->output_shape,
    };

    std::vector<ShapeConstraint> runtime_checks;
    for (const auto& deferred: broadcast_result->deferred_axes) {
        runtime_checks.push_back(ShapeConstraint{
                .condition = DimBroadcastableConstraint{
                        .lhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput,
                                                          .tensor_idx = 0},
                                .dim_index = deferred.lhs_axis,
                        },
                        .rhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = deferred.rhs_axis,
                        },
                },
                .error_context = "SiluMul input dimensions are not broadcastable",
        });
    }

    return InferenceResult{
            .outputs = {std::move(output_spec)},
            .runtime_checks = std::move(runtime_checks),
    };
}

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

}// namespace aethermind
