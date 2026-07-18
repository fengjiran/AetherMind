#include "aethermind/operators/add_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include <span>
#include <string>

namespace aethermind {
namespace {

Status ValidateAddDTypes(const TensorSpec& lhs, const TensorSpec& rhs) {
    if (lhs.dtype != rhs.dtype) {
        return Status::InvalidArgument("Add inputs must have matching dtypes");
    }

    if (!IsAddSupportedDType(lhs.dtype)) {
        return Status::InvalidArgument(
                MakeAddUnsupportedDTypeMessage("Add"));
    }
    return Status::Ok();
}

}// namespace

Status AddOp::ValidateParams() const {
    return Status::Ok();
}

Status AddOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument("Add expects exactly 2 inputs, got " +
                                       std::to_string(inputs.size()));
    }

    const auto& lhs_spec = inputs[0];
    const auto& rhs_spec = inputs[1];

    AM_RETURN_IF_ERROR(ValidateAddDTypes(lhs_spec, rhs_spec));

    if (auto broadcast_result =
                InferBroadcastShape(lhs_spec.shape, rhs_spec.shape);
        !broadcast_result.ok()) {
        return broadcast_result.status();
    }

    return Status::Ok();
}

StatusOr<InferenceResult> AddOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Add expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& lhs_spec = inputs[0];
    const auto& rhs_spec = inputs[1];

    AM_RETURN_IF_ERROR(ValidateAddDTypes(lhs_spec, rhs_spec));

    auto broadcast_result = InferBroadcastShape(
            lhs_spec.shape, rhs_spec.shape);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    TensorSpec output_spec{
            .dtype = lhs_spec.dtype,
            .shape = broadcast_result->output_shape,
    };

    std::vector<ShapeConstraint> runtime_checks;
    for (const auto& deferred: broadcast_result->deferred_axes) {
        runtime_checks.push_back({
                .condition = DimBroadcastableConstraint{
                        .lhs = DimLocator{
                                .tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = deferred.lhs_axis,
                        },
                        .rhs = DimLocator{
                                .tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = deferred.rhs_axis,
                        },
                },
                .error_context = "Add input dimensions are not broadcastable",
        });
    }

    return InferenceResult{
            .outputs = {std::move(output_spec)},
            .runtime_checks = std::move(runtime_checks),
    };
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

AM_REGISTER_OPERATOR(OpType::kAdd, AddOp)

}// namespace aethermind
