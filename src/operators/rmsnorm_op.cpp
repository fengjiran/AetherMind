#include "aethermind/operators/rmsnorm_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

#include <span>
#include <string>

namespace aethermind {
Status RmsNormOp::ValidateParams() const {
    if (params_.eps <= 0.0f) {
        return Status::InvalidArgument(
                "RmsNorm epsilon must be positive, got " + std::to_string(params_.eps));
    }
    return Status::Ok();
}

Status RmsNormOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input_spec = inputs[0];
    const auto& weight_spec = inputs[1];

    if (input_spec.dtype != DataType::Float32() || weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("RmsNorm only supports float32 inputs in Phase 1");
    }

    if (!HasRank(input_spec.shape, 2)) {
        return Status::InvalidArgument("RmsNorm input must be rank-2 [seq_len, hidden]");
    }

    if (!HasRank(weight_spec.shape, 1)) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    const ShapeSymbol& hidden_size = input_spec.shape[1];
    if (!IsPositiveIfStatic(hidden_size)) {
        return Status::InvalidArgument("RmsNorm hidden size must be positive");
    }

    if (!UnifyShapeSymbol(hidden_size, weight_spec.shape[0]).ok()) {
        return Status::InvalidArgument(
                "RmsNorm weight length must equal input last dimension");
    }
    return Status::Ok();
}

StatusOr<InferenceResult> RmsNormOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    if (!HasRank(inputs[0].shape, 2)) {
        return Status::InvalidArgument(
                "RmsNorm input shape must be rank-2 [seq_len, hidden]");
    }

    if (!HasRank(inputs[1].shape, 1)) {
        return Status::InvalidArgument("RmsNorm weight shape must be rank-1");
    }

    if (!IsPositiveIfStatic(inputs[0].shape[1]) || !IsPositiveIfStatic(inputs[1].shape[0])) {
        return Status::InvalidArgument(
                "RmsNorm hidden size and weight length must be positive");
    }

    std::vector<ShapeConstraint> runtime_checks;
    if (inputs[0].shape[1] != inputs[1].shape[0]) {
        runtime_checks.push_back({
                .condition = DimEqualConstraint{
                        .lhs = {
                                .tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = 1,
                        },
                        .rhs = {
                                .tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = 0,
                        }},
                .error_context = "RmsNorm hidden dimension must match weight length",
        });
    }

    return InferenceResult{
            .outputs = {inputs[0]},
            .runtime_checks = std::move(runtime_checks),
    };
}

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

    // Phase 1 CPU-first: construct CPU-specific params directly. Phase 2 should
    // inject params construction via Backend to support multiple backends.
    cpu::CpuRmsNormParams params{
            .input_tensor = b->inputs[0],
            .weight_tensor = b->inputs[1],
            .output_tensor = b->outputs[0],
    };
    ctx.kernel_params = &params;
    return resolved_kernel_.fn(ctx);
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)

}// namespace aethermind
