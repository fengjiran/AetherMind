#ifndef AETHERMIND_OPERATORS_LINEAR_OP_H
#define AETHERMIND_OPERATORS_LINEAR_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Semantic operator for linear transformation: output = input @ weight.T
///
/// Weight shape convention: [out_features, in_features] (PyTorch/HF row-major).
/// Input shape: [..., in_features] (rank 1 or 2 in Phase 1).
/// Output shape: [..., out_features] (same rank as input).
class LinearOp final : public Operator {
public:
    using Params = LinearParams;

    explicit LinearOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kLinear;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Linear";
    }

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

#endif// AETHERMIND_OPERATORS_LINEAR_OP_H
