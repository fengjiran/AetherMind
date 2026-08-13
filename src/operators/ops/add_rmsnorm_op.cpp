#include "aethermind/operators/ops/add_rmsnorm_op.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/ops/rmsnorm_op.h"

#include <cmath>

namespace aethermind {

Status AddRmsNormOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("AddRmsNorm Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kAddRmsNorm,
            ctx.selector);
    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("AddRmsNorm Prepare resolved a kernel with null fn");
    }

    const auto eps_bytes = std::as_bytes(std::span{&params_.eps, size_t{1}});
    resolved_kernel_.attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status AddRmsNormOp::Run(KernelContext& ctx,
                         const RuntimeBindingContext& bindings,
                         size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("AddRmsNorm Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 3U) {
        return Status::InvalidArgument(
                "AddRmsNorm requires 3 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 2U) {
        return Status::InvalidArgument(
                "AddRmsNorm requires 2 output tensor bindings, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kAddRmsNorm, AddRmsNormOp)

namespace detail {
namespace {

Status ValidateExactInputShapes(const TensorSpec& input_spec,
                                const TensorSpec& residual_spec,
                                InferenceResult& result) {
    const std::optional<size_t> input_rank = input_spec.shape.rank();
    const std::optional<size_t> residual_rank = residual_spec.shape.rank();
    if (!input_rank.has_value() || !residual_rank.has_value()) {
        return Status::InvalidArgument("AddRmsNorm input and residual must have ranked shapes");
    }

    if (*input_rank != *residual_rank) {
        return Status::InvalidArgument(
                "AddRmsNorm input and residual must have exactly the same shape; broadcasting is unsupported");
    }

    for (size_t axis = 0; axis < *input_rank; ++axis) {
        const ShapeSymbol& input_dim = input_spec.shape[axis];
        const ShapeSymbol& residual_dim = residual_spec.shape[axis];
        if (AreProvablyEqual(input_dim, residual_dim)) {
            continue;
        }

        if (input_dim.IsStatic() && residual_dim.IsStatic()) {
            return Status::InvalidArgument(
                    "AddRmsNorm input and residual must have exactly the same shape; broadcasting is unsupported");
        }

        result.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = axis},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 1},
                                .dim_index = axis}},
                "AddRmsNorm input and residual dimensions must be equal; broadcasting is unsupported");
    }
    return Status::Ok();
}

}// namespace

StatusOr<InferenceResult> InferAddRmsNorm(const OpParams& params,
                                          std::span<const TensorSpec> inputs) {
    const auto* typed = std::get_if<AddRmsNormParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("AddRmsNorm node requires AddRmsNormParams");
    }

    if (!std::isfinite(typed->eps) || typed->eps <= 0.0F) {
        return Status::InvalidArgument("AddRmsNormParams eps must be finite and positive");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kAddRmsNorm, inputs));

    const TensorSpec& input_spec = inputs[0];
    const TensorSpec& residual_spec = inputs[1];
    const TensorSpec& weight_spec = inputs[2];
    if (input_spec.dtype != residual_spec.dtype) {
        return Status::InvalidArgument("AddRmsNorm input and residual must have the same dtype");
    }

    if (!IsRmsNormSupportedDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeRmsNormUnsupportedDTypeMessage("AddRmsNorm input and residual"));
    }

    if (!IsRmsNormSupportedDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeRmsNormUnsupportedDTypeMessage("AddRmsNorm weight"));
    }

    InferenceResult result;
    AM_RETURN_IF_ERROR(ValidateExactInputShapes(input_spec, residual_spec, result));

    const size_t rank = *input_spec.shape.rank();
    if (rank < 1U) {
        return Status::InvalidArgument("AddRmsNorm input must have rank >= 1");
    }

    const ShapeSymbol& hidden_size = input_spec.shape[rank - 1U];
    if (hidden_size.IsStatic()) {
        if (hidden_size.GetStaticValue() <= 0) {
            return Status::InvalidArgument(
                    "AddRmsNorm input last dimension must be positive when statically known");
        }
    } else {
        result.runtime_checks.emplace_back(
                DimPositiveConstraint{
                        {.tensor_port = {.direction = TensorPortType::kInput,
                                         .tensor_idx = 0},
                         .dim_index = rank - 1U}},
                "AddRmsNorm input last dimension must be positive");
    }

    if (!HasRank(weight_spec.shape, 1U)) {
        return Status::InvalidArgument("AddRmsNorm weight must be rank-1");
    }

    const ShapeSymbol& weight_len = weight_spec.shape[0];
    if (!AreProvablyEqual(hidden_size, weight_len)) {
        if (hidden_size.IsStatic() && weight_len.IsStatic()) {
            return Status::InvalidArgument(
                    "AddRmsNorm weight length must equal input last dimension");
        }

        result.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = rank - 1U},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 2},
                                .dim_index = 0}},
                "AddRmsNorm hidden dimension must match weight length");
    }

    result.outputs.emplace_back(input_spec);
    result.outputs.emplace_back(input_spec);
    return result;
}

}// namespace detail

}// namespace aethermind
