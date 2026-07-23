#ifndef AETHERMIND_OPERATORS_SILU_OP_H
#define AETHERMIND_OPERATORS_SILU_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// SiLU (Swish) element-wise activation operator.
///
/// Computes `output = input * sigmoid(input) = input / (1 + exp(-input))`.
///
/// Phase 1 scope:
/// - Unary element-wise operation (no broadcasting).
/// - float32 only.
/// - Output shape identical to input shape.
///
/// Kernel execution is not yet wired; Run() returns Unimplemented after
/// binding validation. A CPU kernel can be added later without changing the
/// semantic-layer contract.
class SiluOp final : public Operator {
public:
    using Params = SiluParams;

    explicit SiluOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kSilu;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Silu";
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
