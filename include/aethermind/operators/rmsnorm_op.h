// Semantic-layer operator for Root-Mean-Square Layer Normalization.
//
// Inputs (per CheckInputSpecs / InferOutputShapes):
//   - inputs[0]:  rank-2 activations, shape [seq_len, hidden], float32
//   - inputs[1]:  rank-1 learned weight, shape [hidden], float32
// Output: same shape and dtype as inputs[0] (RmsNorm is shape-preserving).
//
// Phase 1 supports only float32; the float32 check is enforced in
// CheckInputSpecs. The `eps` parameter (RmsNormParams.eps) is serialized
// as raw bytes into the resolved kernel's attrs during Prepare() and read
// back by the backend kernel.
//
// Lifecycle and thread-safety follow the Operator base class contract:
// Construct -> ValidateParams() -> Prepare() -> Run() (repeated).
// Instances are not thread-safe; concurrent Run() requires external
// synchronization or one instance per thread.

#ifndef AETHERMIND_OPERATORS_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_RMS_NORM_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Shape-preserving RMS normalization operator.
///
/// Validates that input is rank-2 [seq_len, hidden] float32 and weight is
/// rank-1 [hidden] float32 with matching hidden dimension. On Prepare(),
/// resolves the backend kernel and stores `eps` as raw bytes in
/// `resolved_kernel_.attrs`; on Run(), dispatches to that kernel via
/// `Operator::InvokeResolvedKernel`. No operator-level workspace is
/// required — scratch (e.g. the per-row RMS accumulator) is owned by the
/// backend kernel itself.
class RmsNormOp final : public Operator {
public:
    using Params = RmsNormParams;

    explicit RmsNormOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kRmsNorm;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "RmsNorm";
    }

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override;
    AM_NODISCARD StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override;

    // RmsNorm needs no operator-level workspace: any per-row RMS
    // accumulator is owned by the backend kernel.
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
    // Written only by Prepare(); read by Run() and GetResolvedKernel().
    // `attrs` carries the raw-byte serialization of params_.eps for the
    // backend kernel to read at execution time.
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif