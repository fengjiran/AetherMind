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

// Validates the Linear operator semantics at graph-build time and infers the output shape.
//
// Validation steps (in order):
//  1. Parameter type: params must hold LinearParams (no extra attributes).
//  2. Input count: exactly 2 — [activation, weight].
//  3. Input rank: activation must be rank >= 1 (all dimensions except the last
//     are batch dimensions; the last is in_features).  Weight must be rank 2
//     (matrix).  There is no upper bound on input rank — batched Linear simply
//     broadcasts over all leading dimensions.
//  4. Dimension positivity: all statically-known dimensions must be positive.
//  5. Dtype support: activation and weight dtypes are validated separately
//     because supported quantization schemes (e.g. INT4 weight, FP32 activation)
//     differ between the two.
//  6. Dimension compatibility: input last-dim must equal weight inner-dim
//     (in_features == weight_in).  When both are static and mismatch the
//     graph is invalid (hard error).  When at least one is dynamic a runtime
//     check is emitted instead.
//
// Output shape: input's leading dims + [weight_out].  E.g. [b1, b2, in] →
// [b1, b2, out]; [in] → [out].
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
    const auto& input_shape = input_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_shape.rank();
    if (!input_rank.has_value() || *input_rank < 1) {
        return Status::InvalidArgument("Linear input must have rank >= 1");
    }

    for (const auto dim: input_shape) {
        if (!IsPositiveIfStatic(dim)) {
            return Status::InvalidArgument(
                    "Linear input dimension must be positive when statically known.");
        }
    }

    if (!HasRank(weight_shape, 2)) {
        return Status::InvalidArgument("Linear weight must be rank 2");
    }

    for (const auto dim: weight_shape) {
        if (!IsPositiveIfStatic(dim)) {
            return Status::InvalidArgument(
                    "Linear weight dimension must be positive when statically known.");
        }
    }

    if (!IsLinearSupportedActivationDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedActivationDTypeMessage("Linear"));
    }

    if (!IsLinearSupportedWeightDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeLinearUnsupportedWeightDTypeMessage("Linear"));
    }

    // in_features (input_last_dim) must equal weight_in (weight[1]).
    // Static mismatch → graph is structurally invalid.
    // Dynamic mismatch → defer to a runtime dimension-equality check.
    const ShapeSymbol& in_features = input_shape[*input_rank - 1];
    const ShapeSymbol& weight_in = weight_shape[1];
    std::vector<ShapeConstraint> runtime_checks;
    if (in_features != weight_in) {
        if (in_features.IsStatic() && weight_in.IsStatic()) {
            return Status::InvalidArgument(
                    "Linear weight length must equal input last dimension");
        }

        runtime_checks.push_back({
                .condition = DimEqualConstraint{
                        .lhs = {
                                .tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = *input_rank - 1,
                        },
                        .rhs = {
                                .tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = 1,
                        }},
                .error_context = "Linear input last dimension must match weight input dimension",
        });
    }

    // Output shape = input's leading dims (all except last) + [weight_out].
    // E.g. [b1, b2, in] → [b1, b2, out];  [in] → [out].
    std::vector<ShapeSymbol> output_shape;
    output_shape.reserve(*input_rank);
    for (size_t i = 0; i < *input_rank - 1; ++i) {
        output_shape.push_back(input_shape[i]);
    }
    output_shape.push_back(weight_shape[0]);

    InferenceResult result;
    result.outputs.push_back({input_spec.dtype, SymbolicShape(std::move(output_shape))});
    result.runtime_checks = std::move(runtime_checks);

    return result;
}

}// namespace detail

}// namespace aethermind
