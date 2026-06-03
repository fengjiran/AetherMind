#include "aethermind/operators/rms_norm_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"

#include <span>
#include <string>

namespace aethermind {

Status RmsNormOp::ValidateParams() const {
    if (params_.epsilon <= 0.0f) {
        return Status::InvalidArgument(
                "RmsNorm epsilon must be positive, got " + std::to_string(params_.epsilon));
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

    if (input_spec.dtype_ != DataType::Float32() || weight_spec.dtype_ != DataType::Float32()) {
        return Status::InvalidArgument("RmsNorm only supports float32 inputs in Phase 1");
    }

    if (input_spec.shape_.size() != 2) {
        return Status::InvalidArgument("RmsNorm input must be rank-2 [seq_len, hidden]");
    }

    if (weight_spec.shape_.size() != 1) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    const int64_t hidden_size = input_spec.shape_[1];
    if (hidden_size <= 0) {
        return Status::InvalidArgument("RmsNorm hidden size must be positive");
    }

    if (weight_spec.shape_[0] != hidden_size) {
        return Status::InvalidArgument(
                "RmsNorm weight length must equal input last dimension");
    }
    return Status::Ok();
}

StatusOr<std::vector<TensorSpec>> RmsNormOp::InferOutputShapes(
        std::span<const TensorSpec> inputs) const {
    if (inputs.empty()) {
        return Status::InvalidArgument("RmsNorm requires input shape metadata");
    }
    return std::vector<TensorSpec>{inputs[0]};
}

Status RmsNormOp::Prepare(OperatorContext& ctx) {
    AM_RETURN_IF_ERROR(ValidateParams());
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
    resolved_kernel_.attrs = std::span(
            reinterpret_cast<const std::byte*>(&params_.epsilon), sizeof(float));
    return Status::Ok();
}

Status RmsNormOp::Run(KernelContext& ctx,
                      const RuntimeBindingContext& bindings,
                      size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status(StatusCode::kFailedPrecondition, "RmsNorm Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "RMSNorm requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }
    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "RMSNorm requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    CpuRmsNormParams params{
            .input_tensor = b->inputs[0],
            .weight_tensor = b->inputs[1],
            .output_tensor = b->outputs[0],
    };
    ctx.packed_params = &params;
    return resolved_kernel_.fn(ctx);
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)

}// namespace aethermind
