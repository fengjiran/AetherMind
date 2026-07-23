#ifndef AETHERMIND_OPERATORS_MATMUL_OP_H
#define AETHERMIND_OPERATORS_MATMUL_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Semantic operator for batched matrix multiplication.
///
/// Computes `output = lhs @ effective_rhs` where:
/// - `effective_rhs = rhs` when `transpose_rhs == false` (rhs shape [..., K, N])
/// - `effective_rhs = rhs^T` when `transpose_rhs == true`  (rhs shape [..., N, K])
///
/// Both inputs must have rank >= 2. Trailing two axes are the matrix axes;
/// leading axes are batch axes broadcast according to NumPy semantics.
/// Output shape: `broadcast(lhs_batch, rhs_batch) + [M, N]` where
/// `M = lhs.shape[-2]` and `N = effective_rhs.shape[-1]`.
///
/// Phase 1 scope:
/// - float32 only.
/// - Supports transpose_rhs for PyTorch/HF-style weight storage.
///
/// Kernel execution is not yet wired; Run() returns Unimplemented after
/// binding validation. A CPU kernel can be added later without changing the
/// semantic-layer contract.
class MatMulOp final : public Operator {
public:
    using Params = MatMulParams;

    explicit MatMulOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kMatMul;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "MatMul";
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

#endif// AETHERMIND_OPERATORS_MATMUL_OP_H
