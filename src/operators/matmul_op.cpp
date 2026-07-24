#include "aethermind/operators/matmul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

namespace aethermind {
Status MatMulOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("MatMul Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kMatMul,
            ctx.selector);

    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("MatMul Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status MatMulOp::Run(KernelContext& ctx,
                     const RuntimeBindingContext& bindings,
                     size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("MatMul Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "MatMul requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "MatMul requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    // Kernel not yet implemented. Binding contracts are validated above so
    // that the semantic layer is exercised; execution will be enabled once a
    // CPU kernel is registered and the kernel-params contract is defined.
    return Status::Unimplemented("MatMul kernel not yet implemented");
}

AM_REGISTER_OPERATOR(OpType::kMatMul, MatMulOp)


namespace detail {

namespace {

struct RhsAxes {
    size_t inner;
    size_t outer;
};

RhsAxes ResolveRhsAxes(bool transpose_rhs, size_t rhs_rank) {
    if (transpose_rhs) {
        return {.inner = rhs_rank - 1, .outer = rhs_rank - 2};
    }
    return {.inner = rhs_rank - 2, .outer = rhs_rank - 1};
}


SymbolicShape MakeBatchShape(const SymbolicShape& shape, size_t rank) {
    std::vector<ShapeSymbol> batch_dims;
    for (size_t i = 0; i < rank - 2; ++i) {
        batch_dims.push_back(shape[i]);
    }
    return SymbolicShape(std::move(batch_dims));
}

}// namespace

StatusOr<InferenceResult> InferMatMul(const OpParams& params,
                                      std::span<const TensorSpec> inputs) {
    const auto* matmul_params = std::get_if<MatMulParams>(&params);
    if (matmul_params == nullptr) {
        return Status::InvalidArgument("MatMul node requires MatMulParams");
    }

    if (inputs.size() != 2) {
        return Status::InvalidArgument("MatMul requires exactly 2 inputs");
    }

    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (!IsMatMulSupportedDType(lhs_spec.dtype)) {
        return Status::InvalidArgument(
                MakeMatMulUnsupportedDTypeMessage("MatMul lhs"));
    }

    if (!IsMatMulSupportedDType(rhs_spec.dtype)) {
        return Status::InvalidArgument(
                MakeMatMulUnsupportedDTypeMessage("MatMul rhs"));
    }

    const auto& lhs_shape = lhs_spec.shape;
    const auto& rhs_shape = rhs_spec.shape;
    const auto lhs_rank = lhs_shape.rank();
    const auto rhs_rank = rhs_shape.rank();
    if (!lhs_rank.has_value() || *lhs_rank < 2) {
        return Status::InvalidArgument("MatMul lhs must have rank >= 2");
    }

    if (!rhs_rank.has_value() || *rhs_rank < 2) {
        return Status::InvalidArgument("MatMul rhs must have rank >= 2");
    }

    const RhsAxes rhs_axes = ResolveRhsAxes(matmul_params->transpose_rhs, *rhs_rank);
    const ShapeSymbol& lhs_inner_dim = lhs_spec.shape[*lhs_rank - 1];
    const ShapeSymbol& rhs_inner_dim = rhs_spec.shape[rhs_axes.inner];

    // Zero-valued M/N/K and batch dimensions are intentionally allowed,
    // matching NumPy/PyTorch semantics (M=0/N=0 -> empty output, K=0 ->
    // zero-valued output). Model-weight positivity is enforced at the
    // GraphOpBuilder layer, not by this generic operator.
    const ShapeSymbol& lhs_outer = lhs_spec.shape[*lhs_rank - 2];
    const ShapeSymbol& rhs_outer = rhs_spec.shape[rhs_axes.outer];

    auto lhs_batch = MakeBatchShape(lhs_spec.shape, *lhs_rank);
    auto rhs_batch = MakeBatchShape(rhs_spec.shape, *rhs_rank);
    auto broadcast_result = InferBroadcastShape(lhs_batch, rhs_batch);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    std::vector<ShapeSymbol> output_shape = broadcast_result->output_shape.shape().value();
    output_shape.push_back(lhs_outer);
    output_shape.push_back(rhs_outer);

    InferenceResult res;
    res.outputs.emplace_back(lhs_spec.dtype, SymbolicShape(output_shape));
    if (!AreProvablyEqual(lhs_inner_dim, rhs_inner_dim)) {
        if (lhs_inner_dim.IsStatic() && rhs_inner_dim.IsStatic()) {
            return Status::InvalidArgument("MatMul inner dimensions must be equal");
        }

        res.runtime_checks.emplace_back(
                DimEqualConstraint{
                        .lhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 0},
                                .dim_index = *lhs_rank - 1},
                        .rhs = {.tensor_port = {.direction = TensorPortType::kInput,
                                                .tensor_idx = 1},
                                .dim_index = rhs_axes.inner}},
                "MatMul inner dimensions must be equal");
    }

    for (const auto& deferred: broadcast_result->deferred_axes) {
        res.runtime_checks.emplace_back(
                DimBroadcastableConstraint{
                        .lhs = {.tensor_port = {TensorPortType::kInput, 0},
                                .dim_index = deferred.lhs_axis},
                        .rhs = {.tensor_port = {TensorPortType::kInput, 1},
                                .dim_index = deferred.rhs_axis}},
                "MatMul batch dimensions must be broadcastable");
    }
    return res;
}

}// namespace detail

}// namespace aethermind
