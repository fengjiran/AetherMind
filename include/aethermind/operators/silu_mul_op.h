#ifndef AETHERMIND_OPERATORS_SILU_MUL_OP_H
#define AETHERMIND_OPERATORS_SILU_MUL_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Fused SiLU-multiplication operator.
///
/// Computes `output = silu(gate) * up` where `silu(x) = x / (1 + exp(-x))`.
/// Inputs `gate` and `up` are broadcast according to NumPy semantics.
///
/// Phase 1 scope:
/// - Binary element-wise operation with broadcasting.
/// - float32 only.
///
/// Kernel execution is not yet wired; Run() returns Unimplemented after
/// binding validation. A CPU kernel can be added later without changing the
/// semantic-layer contract.
class SiluMulOp final : public Operator {
public:
    using Params = SiluMulParams;

    explicit SiluMulOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kSiluMul;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "SiluMul";
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
