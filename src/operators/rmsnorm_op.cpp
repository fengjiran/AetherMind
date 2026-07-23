#include "aethermind/operators/rmsnorm_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"

#include <cmath>

namespace aethermind {
Status RmsNormOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("RmsNorm Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kRmsNorm,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("RmsNorm Prepare resolved a kernel with null fn");
    }
    const auto eps_bytes = std::as_bytes(std::span{&params_.eps, size_t{1}});
    resolved_kernel_.attrs.assign(eps_bytes.begin(), eps_bytes.end());
    return Status::Ok();
}

Status RmsNormOp::Run(KernelContext& ctx,
                      const RuntimeBindingContext& bindings,
                      size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("RmsNorm Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "RmsNorm requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)


namespace detail {

StatusOr<InferenceResult> InferRmsNorm(const OpParams& params,
                                       std::span<const TensorSpec> inputs) {
    const auto* typed = std::get_if<RmsNormParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("RmsNorm node requires RmsNormParams");
    }

    if (!std::isfinite(typed->eps) || typed->eps <= 0.0F) {
        return Status::InvalidArgument("RmsNormParams eps must be finite and positive");
    }

    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];
    const auto input_rank = input_spec.shape.rank();
    if (!input_rank.has_value() || *input_rank < 1) {
        return Status::InvalidArgument("RmsNorm input must have rank >= 1");
    }

    // Batch dimensions must be positive when statically known.
    for (const auto dim: input_spec.shape) {
        if (!IsPositiveIfStatic(dim)) {
            return Status::InvalidArgument(
                    "RmsNorm input dimension must be positive when statically known.");
        }
    }

    if (!HasRank(weight_spec.shape, 1)) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    const size_t rank = *input_rank;
    const ShapeSymbol& hidden_size = input_spec.shape[rank - 1];
    const ShapeSymbol& weight_len = weight_spec.shape[0];
    if (!IsPositiveIfStatic(weight_len)) {
        return Status::InvalidArgument(
                "RmsNorm weight length must be positive when statically known.");
    }

    std::vector<ShapeConstraint> runtime_checks;
    // Symbols differ — fail-fast if statically known, otherwise defer to runtime.
    if (hidden_size != weight_len) {
        if (hidden_size.IsStatic() && weight_len.IsStatic()) {
            return Status::InvalidArgument(
                    "RmsNorm weight length must equal input last dimension");
        }

        runtime_checks.push_back({
                .condition = DimEqualConstraint{
                        .lhs = {
                                .tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = rank - 1,
                        },
                        .rhs = {
                                .tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = 0,
                        }},
                .error_context = "RmsNorm hidden dimension must match weight length",
        });
    }

    if (!IsRmsNormSupportedDType(input_spec.dtype)) {
        return Status::InvalidArgument(
                MakeRmsNormUnsupportedDTypeMessage("RmsNorm input"));
    }

    if (!IsRmsNormSupportedDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeRmsNormUnsupportedDTypeMessage("RmsNorm weight"));
    }

    return InferenceResult{
            .outputs = {input_spec},
            .runtime_checks = std::move(runtime_checks),
    };
}

}// namespace detail

}// namespace aethermind
