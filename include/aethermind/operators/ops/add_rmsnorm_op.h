#ifndef AETHERMIND_OPERATORS_OPS_ADD_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_OPS_ADD_RMS_NORM_OP_H

/// @file add_rmsnorm_op.h
/// @brief Fused residual addition and RMS normalization operator declaration.

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Fused `RmsNorm(input + residual, weight, eps)` operator.
///
/// The three inputs are ordered as `[input, residual, weight]`. The first
/// output is the normalized activation and the second is the unnormalized
/// residual sum. The operator owns no workspace: any scratch storage belongs
/// to the backend kernel selected during `Prepare()`.
class AddRmsNormOp final : public Operator {
public:
    using Params = AddRmsNormParams;

    explicit AddRmsNormOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kAddRmsNorm;
    }

    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const TensorSpec> inputs) const noexcept override {
        UNUSED(inputs);
        return {};
    }

    Status Prepare(OperatorContext& ctx) override;

    Status Run(KernelContext& ctx,
               const RuntimeBindingContext& bindings,
               size_t step_index) const noexcept override;

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    // `attrs` holds the raw-byte serialization of params_.eps for the backend
    // kernel selected during Prepare().
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
