#include "aethermind/operators/linear_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/shape_constraint.h"

namespace aethermind {

Status LinearOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("Linear Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kLinear,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("Linear Prepare resolved a kernel with null fn");
    }
    // LinearParams is empty; no attrs to write.
    return Status::Ok();
}

Status LinearOp::Run(KernelContext& ctx,
                     const RuntimeBindingContext& bindings,
                     size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("Linear Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "Linear requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "Linear requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kLinear, LinearOp)


namespace detail {

StatusOr<InferenceResult> InferLinear(const OpParams& params,
                                      std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<LinearParams>(params)) {
        return Status::InvalidArgument("Linear node requires LinearParams");
    }

    if (inputs.size() != 2) {
        return Status::InvalidArgument("Linear requires exactly 2 inputs");
    }

    const TensorSpec& input_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    if (input_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Linear input must be float32");
    }
    if (weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Linear weight must be float32");
    }
    if (!input_spec.shape.IsRanked()) {
        return Status::InvalidArgument("Linear input must be ranked");
    }
    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Linear weight must be rank 2");
    }
    const auto& input_shape = input_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_shape.rank().value();
    if (input_rank != 1 && input_rank != 2) {
        return Status::InvalidArgument("Linear input must be rank 1 or 2");
    }
    const ShapeSymbol& in_features = input_shape[input_rank - 1];
    const ShapeSymbol& weight_in = weight_shape[1];
    if (in_features.IsStatic() && in_features.GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear input last dimension must be positive");
    }
    if (weight_shape[0].IsStatic() && weight_shape[0].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear weight output dimension must be positive");
    }
    if (weight_in.IsStatic() && weight_in.GetStaticValue() <= 0) {
        return Status::InvalidArgument("Linear weight input dimension must be positive");
    }
    if (in_features.IsStatic() && weight_in.IsStatic() &&
        in_features.GetStaticValue() != weight_in.GetStaticValue()) {
        return Status::InvalidArgument("Linear input last dimension must match weight input dimension");
    }
    auto unified = UnifyShapeSymbol(in_features, weight_in);
    if (!unified.ok()) {
        return unified.status();
    }
    InferenceResult result;
    if (input_rank == 1) {
        result.outputs.push_back({input_spec.dtype, SymbolicShape({weight_shape[0]})});
    } else {
        result.outputs.push_back({input_spec.dtype, SymbolicShape({input_shape[0], weight_shape[0]})});
    }
    if (in_features != weight_in) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimEqualConstraint{{{TensorPortType::kInput, 0}, static_cast<size_t>(input_rank - 1)},
                                   {{TensorPortType::kInput, 1}, 1}},
                "Linear input last dimension must match weight input dimension"});
    }
    return result;
}

}// namespace detail

}// namespace aethermind
