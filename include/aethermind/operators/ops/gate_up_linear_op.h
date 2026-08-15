#ifndef AETHERMIND_OPERATORS_OPS_GATE_UP_LINEAR_OP_H
#define AETHERMIND_OPERATORS_OPS_GATE_UP_LINEAR_OP_H

/// @file gate_up_linear_op.h
/// @brief Fused MLP gate/up linear-projection semantics and operator declaration.

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Semantic operator for one fused MLP gate/up projection.
///
/// `gate = input @ gate_up_weight[0:gate_out, :].T`; `up` consumes the
/// following `up_out` rows. The packed weight is `[gate_out + up_out,
/// in_features]`, ordered Gate then Up. Its two outputs intentionally keep the
/// existing SiluMul contract intact. Bias inputs are not in the current schema.
class GateUpLinearOp final : public Operator {
public:
    using Params = GateUpLinearParams;

    explicit GateUpLinearOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kGateUpLinear;
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
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
