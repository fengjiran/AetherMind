#ifndef AETHERMIND_OPERATORS_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_RMS_NORM_OP_H

#include "aethermind/operators/operator.h"

#include <array>
#include <cstddef>

namespace aethermind {

struct RmsNormOpParams {
    float epsilon_ = 1.0e-5F;
};

class RmsNormOp final : public Operator {
public:
    using Params = RmsNormOpParams;

    explicit RmsNormOp(RmsNormOpParams params) noexcept;

    AM_NODISCARD OpType Type() const noexcept override;
    AM_NODISCARD const char* Name() const noexcept override;
    AM_NODISCARD Status Validate() const override;
    AM_NODISCARD Status ValidateInputs(std::span<const TensorView> inputs) const override;
    AM_NODISCARD StatusOr<std::vector<ShapeInfo>> InferOutputShapes(
            std::span<const ShapeInfo> inputs) const override;
    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const ShapeInfo> inputs) const noexcept override;
    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;
    AM_NODISCARD Status Run(const KernelInvocation& invocation,
                            const OpKernelContext& op_ctx,
                            const WorkspaceBinding& workspace) const noexcept override;
    AM_NODISCARD ResolvedKernel GetResolvedKernel() const noexcept override;

private:
    RmsNormOpParams params_{};
    alignas(float) std::array<std::byte, sizeof(float)> attrs_{};
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
