#include "aethermind/operators/elementwise_mul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/kernels/cpu_elementwise_mul_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include <span>
#include <string>

namespace aethermind {

Status ElementwiseMulOp::ValidateParams() const {
    return Status::Ok();
}

Status ElementwiseMulOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "ElementwiseMul expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& lhs_spec = inputs[0];
    const auto& rhs_spec = inputs[1];

    if (lhs_spec.dtype != DataType::Float32() || rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("ElementwiseMul only supports float32 inputs in Phase 1");
    }

    auto broadcast_result = InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    return Status::Ok();
}

StatusOr<InferenceResult> ElementwiseMulOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "ElementwiseMul expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& lhs_spec = inputs[0];
    const auto& rhs_spec = inputs[1];

    if (lhs_spec.dtype != DataType::Float32() || rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("ElementwiseMul only supports float32 inputs in Phase 1");
    }

    auto broadcast_result = InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
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
                .error_context = "ElementwiseMul input dimensions are not broadcastable",
        });
    }

    return InferenceResult{
            .outputs = {std::move(output_spec)},
            .runtime_checks = std::move(runtime_checks),
    };
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

    CpuElementwiseMulParams params{
            .lhs_tensor = b->inputs[0],
            .rhs_tensor = b->inputs[1],
            .output_tensor = b->outputs[0],
    };
    ctx.kernel_params = &params;
    return resolved_kernel_.fn(ctx);
}

AM_REGISTER_OPERATOR(OpType::kElementwiseMul, ElementwiseMulOp)

}// namespace aethermind
