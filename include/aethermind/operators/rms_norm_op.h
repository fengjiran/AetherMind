#ifndef AETHERMIND_OPERATORS_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_RMS_NORM_OP_H

#include "aethermind/operators/operator.h"

namespace aethermind {

class RmsNormOp final : public Operator {
public:
    struct Params {
        float epsilon_ = 1.0e-5F;
    };

    explicit RmsNormOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kRmsNorm;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "RmsNorm";
    }

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckShapes(std::span<const ShapeInfo> inputs) const override;
    AM_NODISCARD StatusOr<std::vector<ShapeInfo>> InferOutputShapes(
            std::span<const ShapeInfo> inputs) const override;

    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const ShapeInfo> inputs) const noexcept override {
        UNUSED(inputs);
        return {};
    }

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;

    AM_NODISCARD Status Run(const KernelContext& ctx) const noexcept override {
        if (resolved_kernel_.fn == nullptr) {
            return Status(StatusCode::kFailedPrecondition, "RmsNorm Run called before Prepare");
        }
        return resolved_kernel_.fn(ctx);
    }

    AM_NODISCARD ResolvedKernel GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
