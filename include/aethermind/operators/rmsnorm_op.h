#ifndef AETHERMIND_OPERATORS_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_RMS_NORM_OP_H

#include "aethermind/operators/operator.h"

namespace aethermind {

class RmsNormOp final : public Operator {
public:
    struct Params {
        float eps = 1.0e-5f;
    };

    explicit RmsNormOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kRmsNorm;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "RmsNorm";
    }

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override;
    AM_NODISCARD StatusOr<std::vector<TensorSpec>> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override;

    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const TensorSpec> inputs) const noexcept override {
        UNUSED(inputs);
        return {};
    }

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;

    AM_NODISCARD Status Run(KernelContext& ctx,
                            const RuntimeBindingContext& bindings,
                            size_t step_index) const noexcept override;

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
