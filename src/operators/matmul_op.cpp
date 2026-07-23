#include "aethermind/operators/matmul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include <span>
#include <string>
#include <vector>

namespace aethermind {
Status MatMulOp::ValidateParams() const {
    return ValidateOperatorParams(Type(), params_);
}

Status MatMulOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    return InferOperator(Type(), params_, inputs).status();
}

StatusOr<InferenceResult> MatMulOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    return InferOperator(Type(), params_, inputs);
}

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

RhsAxes ResolveRhsAxes(const MatMulParams& params, size_t rhs_rank) {
    if (params.transpose_rhs) {
        return {rhs_rank - 1, rhs_rank - 2};
    }
    return {rhs_rank - 2, rhs_rank - 1};
}

SymbolicShape MakeBatchShape(const SymbolicShape& shape, size_t rank) {
    std::vector<ShapeSymbol> batch_dims;
    for (size_t i = 0; i < rank - 2; ++i) {
        batch_dims.push_back(shape[i]);
    }
    return SymbolicShape(batch_dims);
}

}// namespace

StatusOr<InferenceResult> InferMatMul(const OpParams& params,
                                      std::span<const TensorSpec> inputs) {
    if (inputs.size() != 2) {
        return Status::InvalidArgument("MatMul requires exactly 2 inputs");
    }
    const auto* typed = std::get_if<MatMulParams>(&params);
    if (typed == nullptr) {
        return Status::InvalidArgument("MatMul requires MatMulParams");
    }
    const TensorSpec& lhs_spec = inputs[0];
    const TensorSpec& rhs_spec = inputs[1];
    if (lhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("MatMul lhs must be float32");
    }
    if (rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("MatMul rhs must be float32");
    }
    if (!lhs_spec.shape.IsRanked()) {
        return Status::InvalidArgument("MatMul lhs must be ranked");
    }
    if (!rhs_spec.shape.IsRanked()) {
        return Status::InvalidArgument("MatMul rhs must be ranked");
    }
    const auto lhs_rank = lhs_spec.shape.rank().value();
    const auto rhs_rank = rhs_spec.shape.rank().value();
    if (lhs_rank < 2) {
        return Status::InvalidArgument("MatMul lhs must have rank >= 2");
    }
    if (rhs_rank < 2) {
        return Status::InvalidArgument("MatMul rhs must have rank >= 2");
    }
    const RhsAxes rhs_axes = ResolveRhsAxes(*typed, rhs_rank);
    const ShapeSymbol& lhs_inner = lhs_spec.shape[lhs_rank - 1];
    const ShapeSymbol& rhs_inner = rhs_spec.shape[rhs_axes.inner];
    if (lhs_inner.IsStatic() && lhs_inner.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul lhs inner dimension must be positive");
    }
    if (rhs_inner.IsStatic() && rhs_inner.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul rhs inner dimension must be positive");
    }
    if (lhs_inner.IsStatic() && rhs_inner.IsStatic() &&
        lhs_inner.GetStaticValue() != rhs_inner.GetStaticValue()) {
        return Status::InvalidArgument("MatMul inner dimensions must be equal");
    }
    const ShapeSymbol& lhs_outer = lhs_spec.shape[lhs_rank - 2];
    const ShapeSymbol& rhs_outer = rhs_spec.shape[rhs_axes.outer];
    if (lhs_outer.IsStatic() && lhs_outer.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul lhs outer dimension must be positive");
    }
    if (rhs_outer.IsStatic() && rhs_outer.GetStaticValue() <= 0) {
        return Status::InvalidArgument("MatMul rhs outer dimension must be positive");
    }
    auto lhs_batch = MakeBatchShape(lhs_spec.shape, lhs_rank);
    auto rhs_batch = MakeBatchShape(rhs_spec.shape, rhs_rank);
    auto broadcast_result = InferBroadcastShape(lhs_batch, rhs_batch);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }
    std::vector<ShapeSymbol> output_dims = broadcast_result->output_shape.shape().value();
    output_dims.push_back(lhs_outer);
    output_dims.push_back(rhs_outer);
    InferenceResult result;
    result.outputs.push_back({lhs_spec.dtype, SymbolicShape(output_dims)});
    if (!lhs_inner.IsStatic() || !rhs_inner.IsStatic()) {
        if (lhs_inner != rhs_inner) {
            result.runtime_checks.push_back(ShapeConstraint{
                    DimEqualConstraint{{{TensorPortType::kInput, 0}, lhs_rank - 1},
                                       {{TensorPortType::kInput, 1}, rhs_axes.inner}},
                    "MatMul inner dimensions must be equal"});
        }
    }
    for (const auto& deferred: broadcast_result->deferred_axes) {
        result.runtime_checks.push_back(ShapeConstraint{
                DimBroadcastableConstraint{{{TensorPortType::kInput, 0}, deferred.lhs_axis},
                                           {{TensorPortType::kInput, 1}, deferred.rhs_axis}},
                "MatMul batch dimensions must be broadcastable"});
    }
    return result;
}

}// namespace detail

}// namespace aethermind
