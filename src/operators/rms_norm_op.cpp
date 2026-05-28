#include "aethermind/operators/rms_norm_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/operators/operator_registry.h"

#include <span>
#include <string>

namespace aethermind {

Status RmsNormOp::ValidateParams() const {
    if (params_.epsilon_ <= 0.0F) {
        return Status::InvalidArgument(
                "RmsNorm epsilon must be positive, got " + std::to_string(params_.epsilon_));
    }
    return Status::Ok();
}

Status RmsNormOp::CheckShapes(std::span<const ShapeInfo> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "RmsNorm expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    if (input.dtype_ != DataType::Float32() || weight.dtype_ != DataType::Float32()) {
        return Status::InvalidArgument("RmsNorm only supports float32 inputs in Phase 1");
    }

    if (input.shape_.size() < 1) {
        return Status::InvalidArgument("RmsNorm input rank must be at least 1");
    }

    if (weight.shape_.size() != 1) {
        return Status::InvalidArgument("RmsNorm weight must be rank-1");
    }

    const int64_t hidden_size = input.shape_.back();
    if (hidden_size <= 0) {
        return Status::InvalidArgument("RmsNorm hidden size must be positive");
    }

    if (weight.shape_[0] != hidden_size) {
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
            reinterpret_cast<const std::byte*>(&params_.epsilon_), sizeof(float));
    return Status::Ok();
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)

}// namespace aethermind
