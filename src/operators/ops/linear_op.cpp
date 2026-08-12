#include "aethermind/operators/ops/linear_op.h"
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

// Zero-valued dimensions are intentionally allowed, matching NumPy/PyTorch
// semantics (Linear is semantically equivalent to MatMul). Model-weight
// positivity is enforced at the GraphOpBuilder layer.
StatusOr<InferenceResult> InferLinear(const OpParams& params,
                                      std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<LinearParams>(params)) {
        return Status::InvalidArgument("Linear node requires LinearParams");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kLinear, inputs));

    const TensorSpec& input_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    const auto& input_shape = input_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_shape.rank();
    if (!input_rank.has_value() || *input_rank < 1) {
        return Status::InvalidArgument("Linear input must have rank >= 1");
    }

    if (!HasRank(weight_shape, 2)) {
        return Status::InvalidArgument("Linear weight must be rank 2");
    }

    if (!IsLinearSupportedActivationDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedActivationDTypeMessage("Linear"));
    }

    if (!IsLinearSupportedWeightDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedWeightDTypeMessage("Linear"));
    }

    // Reject proven incompatibility now; preserve symbolic compatibility as a
    // runtime equality check.
    const ShapeSymbol& in_features = input_shape[*input_rank - 1];
    const ShapeSymbol& out_features = weight_shape[0];
    const ShapeSymbol& weight_in = weight_shape[1];

    InferenceResult res;
    if (!AreProvablyEqual(in_features, weight_in)) {
        if (in_features.IsStatic() && weight_in.IsStatic()) {
            return Status::InvalidArgument(
                    "Linear weight length must equal input last dimension");
        }

        res.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = *input_rank - 1},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 1},
                                .dim_index = 1}},
                "Linear input last dimension must match weight input dimension");
    }

    std::vector<ShapeSymbol> output_shape;
    output_shape.reserve(*input_rank);
    for (size_t i = 0; i < *input_rank - 1; ++i) {
        output_shape.push_back(input_shape[i]);
    }
    output_shape.push_back(out_features);

    res.outputs.emplace_back(input_spec.dtype, SymbolicShape(std::move(output_shape)));

    return res;
}

}// namespace detail

}// namespace aethermind
