#include "aethermind/operators/rms_norm_op.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/operators/operator_registry.h"

#include <cstring>
#include <span>
#include <string>

namespace aethermind {

static_assert(sizeof(CpuRmsNormAttrs) == sizeof(float));
static_assert(alignof(CpuRmsNormAttrs) <= alignof(float));

RmsNormOp::RmsNormOp(RmsNormOpParams params) noexcept : params_(params) {}

OpType RmsNormOp::Type() const noexcept {
    return OpType::kRmsNorm;
}

const char* RmsNormOp::Name() const noexcept {
    return "RmsNorm";
}

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

    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
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

WorkspaceRequirement RmsNormOp::ComputeWorkspaceRequirement(
        std::span<const ShapeInfo> inputs) const noexcept {
    (void) inputs;
    return {};
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

    const CpuRmsNormAttrs attrs{.Epsilon = params_.epsilon_};
    std::memcpy(attrs_.data(), &attrs, sizeof(attrs));

    resolved_kernel_ = resolved.value();
    resolved_kernel_.attrs = std::span<const std::byte>(attrs_.data(), attrs_.size());
    return Status::Ok();
}

Status RmsNormOp::Run(const KernelInvocation& invocation,
                      const OpKernelContext& op_ctx,
                      const WorkspaceBinding& workspace) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status(StatusCode::kFailedPrecondition, "RmsNorm Run called before Prepare");
    }
    return resolved_kernel_.fn(invocation, op_ctx, workspace);
}

ResolvedKernel RmsNormOp::GetResolvedKernel() const noexcept {
    return resolved_kernel_;
}

AM_REGISTER_OPERATOR(OpType::kRmsNorm, RmsNormOp)

}// namespace aethermind
