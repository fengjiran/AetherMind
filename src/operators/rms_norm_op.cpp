#include "aethermind/operators/rms_norm_op.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/operators/operator_registry.h"

#include <span>
#include <string>

namespace aethermind {

Status RmsNormOp::Validate() const {
    if (params_.epsilon_ <= 0.0F) {
        return Status::InvalidArgument(
                "RmsNorm epsilon must be positive, got " + std::to_string(params_.epsilon_));
    }
    return Status::Ok();
}

Status RmsNormOp::ValidateInputs(std::span<const TensorView> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    if (!input.is_valid()) {
        return Status::InvalidArgument("RmsNorm input TensorView must be valid");
    }

    if (!weight.is_valid()) {
        return Status::InvalidArgument("RmsNorm weight TensorView must be valid");
    }

    if (input.dtype() != DataType::Float32() || weight.dtype() != DataType::Float32()) {
        return Status::InvalidArgument("RmsNorm only supports float32 inputs in Phase 1");
    }

    if (input.rank() < 1) {
        return Status::InvalidArgument("RmsNorm input rank must be at least 1");
    }

    if (weight.rank() != 1) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    if (!input.is_contiguous() || !weight.is_contiguous()) {
        return Status::InvalidArgument("RmsNorm requires contiguous input and weight");
    }

    const int64_t hidden_size = input.dim(input.rank() - 1);
    if (hidden_size <= 0) {
        return Status::InvalidArgument("RmsNorm hidden size must be positive");
    }

    if (weight.numel() != hidden_size) {
        return Status::InvalidArgument(
                "RmsNorm weight length must equal input last dimension");
    }
    return Status::Ok();
}

StatusOr<std::vector<ShapeInfo>> RmsNormOp::InferOutputShapes(
        std::span<const ShapeInfo> inputs) const {
    if (inputs.empty()) {
        return Status::InvalidArgument("RmsNorm requires input shape metadata");
    }
    return std::vector<ShapeInfo>{inputs[0]};
}

Status RmsNormOp::Prepare(OperatorContext& ctx) {
    AM_RETURN_IF_ERROR(Validate());
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("RmsNorm Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(KernelRequest{
            .op_type = OpType::kRmsNorm,
            .selector = ctx.selector,
    });
    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    resolved_kernel_.attrs = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(&params_.epsilon_), sizeof(float));
    return Status::Ok();
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)

}// namespace aethermind
