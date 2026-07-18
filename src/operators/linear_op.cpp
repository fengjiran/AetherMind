#include "aethermind/operators/linear_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"

#include <span>
#include <string>
#include <vector>

namespace aethermind {

Status LinearOp::ValidateParams() const {
    // LinearParams is empty in Phase 1; nothing to validate.
    return Status::Ok();
}

Status LinearOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Linear expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];

    if (input_spec.dtype != DataType::Float32() || weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Linear only supports float32 in Phase 1");
    }

    // Phase 1 supports input rank 1 or 2.
    const auto input_rank_opt = input_spec.shape.rank();
    if (!input_rank_opt.has_value() || *input_rank_opt < 1 || *input_rank_opt > 2) {
        return Status::InvalidArgument("Linear input must have rank 1 or 2 in Phase 1");
    }

    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Linear weight must be rank-2 [out_features, in_features]");
    }

    const ShapeSymbol& in_features = input_spec.shape[*input_rank_opt - 1];
    if (!IsPositiveIfStatic(in_features)) {
        return Status::InvalidArgument("Linear input last dimension must be positive");
    }

    if (!UnifyShapeSymbol(in_features, weight_spec.shape[1]).ok()) {
        return Status::InvalidArgument(
                "Linear input last dimension must equal weight in_features");
    }

    return Status::Ok();
}

StatusOr<InferenceResult> LinearOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Linear expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];

    const auto input_rank_opt = input_spec.shape.rank();
    if (!input_rank_opt.has_value() || *input_rank_opt < 1) {
        return Status::InvalidArgument("Linear input shape must have rank >= 1");
    }

    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Linear weight shape must be rank-2");
    }

    const size_t input_rank = *input_rank_opt;
    const ShapeSymbol& in_features = input_spec.shape[input_rank - 1];
    const ShapeSymbol& weight_in = weight_spec.shape[1];

    if (!IsPositiveIfStatic(in_features) || !IsPositiveIfStatic(weight_in)) {
        return Status::InvalidArgument(
                "Linear in_features and weight in_features must be positive");
    }

    // output shape = input.shape[:-1] + [weight.shape[0]]
    std::vector<ShapeSymbol> output_dims;
    output_dims.reserve(input_rank);
    for (size_t i = 0; i < input_rank - 1; ++i) {
        output_dims.push_back(input_spec.shape[i]);
    }
    output_dims.push_back(weight_spec.shape[0]);

    TensorSpec output_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape(std::move(output_dims)),
    };

    InferenceResult result;
    result.outputs.push_back(std::move(output_spec));

    // Emit runtime check when in_features cannot be proven equal statically.
    if (in_features != weight_in) {
        result.runtime_checks.push_back(ShapeConstraint{
                .condition = DimEqualConstraint{
                        .lhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput,
                                                          .tensor_idx = 0},
                                .dim_index = input_rank - 1,
                        },
                        .rhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = 1,
                        }},
                .error_context = "Linear input last dimension must equal weight in_features",
        });
    }

    return result;
}

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

}// namespace aethermind
