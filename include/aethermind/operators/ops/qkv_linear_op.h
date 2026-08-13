#ifndef AETHERMIND_OPERATORS_OPS_QKV_LINEAR_OP_H
#define AETHERMIND_OPERATORS_OPS_QKV_LINEAR_OP_H

/// @file qkv_linear_op.h
/// @brief Fused Q/K/V linear-projection semantics and executable operator declaration.

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Semantic operator for a single fused Q/K/V linear projection.
///
/// `output_q = input @ qkv_weight[0:q_out, :].T`, `output_k` and `output_v`
/// slice the following `k_out` / `v_out` packed rows respectively. The packed
/// weight convention follows Linear: [q_out + k_out + v_out, in_features]
/// (PyTorch/HF row-major), with rows ordered Q, then K, then V.
///
/// Input shape: [..., in_features] with rank at least one.
/// Output shapes: q [..., q_out], k [..., k_out], v [..., v_out].
///
/// Dtype contract mirrors Linear (see linear_op.h): the activation dtype set
/// and the weight dtype set are shared, and the output dtype follows the
/// activation dtype. Bias inputs are not supported by the current two-port
/// schema; `QkvLinearParams::has_bias` must be false.
class QkvLinearOp final : public Operator {
public:
    using Params = QkvLinearParams;

    explicit QkvLinearOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kQkvLinear;
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

#endif// AETHERMIND_OPERATORS_OPS_QKV_LINEAR_OP_H
