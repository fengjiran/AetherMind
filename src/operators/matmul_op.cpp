#include "aethermind/operators/matmul_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/shape_inference/broadcast.h"

#include <span>
#include <string>
#include <vector>

namespace aethermind {
namespace {

// Locates the contraction (K) and output-column (N) axes of rhs, accounting
// for transpose_rhs:
// - transpose_rhs=false: rhs layout is [..., K, N]
// - transpose_rhs=true:  rhs layout is [..., N, K]
struct EffectiveRhsAxes {
    size_t inner_idx;// K axis (must equal lhs.shape[-1])
    size_t outer_idx;// N axis (becomes output last dim)
};

EffectiveRhsAxes ResolveRhsAxes(size_t rhs_rank, bool transpose_rhs) noexcept {
    if (transpose_rhs) {
        return {.inner_idx = rhs_rank - 1, .outer_idx = rhs_rank - 2};
    }
    return {.inner_idx = rhs_rank - 2, .outer_idx = rhs_rank - 1};
}

// Returns the leading batch axes of `shape` (i.e. shape[:-2]).
// Caller must ensure `shape` is ranked with rank >= 2.
SymbolicShape MakeBatchShape(const SymbolicShape& shape, size_t rank) {
    std::vector<ShapeSymbol> batch_dims;
    batch_dims.reserve(rank - 2);
    for (size_t i = 0; i < rank - 2; ++i) {
        batch_dims.push_back(shape[i]);
    }
    return SymbolicShape(std::move(batch_dims));
}

}// namespace

Status MatMulOp::ValidateParams() const {
    // MatMulParams has a single bool flag with no invalid value.
    return Status::Ok();
}

Status MatMulOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "MatMul expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& lhs_spec = inputs[0];
    const auto& rhs_spec = inputs[1];

    if (lhs_spec.dtype != DataType::Float32() || rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("MatMul only supports float32 inputs in Phase 1");
    }

    const auto lhs_rank_opt = lhs_spec.shape.rank();
    if (!lhs_rank_opt.has_value() || *lhs_rank_opt < 2) {
        return Status::InvalidArgument("MatMul lhs must have rank >= 2");
    }

    const auto rhs_rank_opt = rhs_spec.shape.rank();
    if (!rhs_rank_opt.has_value() || *rhs_rank_opt < 2) {
        return Status::InvalidArgument("MatMul rhs must have rank >= 2");
    }

    const size_t lhs_rank = *lhs_rank_opt;
    const size_t rhs_rank = *rhs_rank_opt;
    const auto rhs_axes = ResolveRhsAxes(rhs_rank, params_.transpose_rhs);

    const ShapeSymbol& lhs_inner = lhs_spec.shape[lhs_rank - 1];
    const ShapeSymbol& rhs_inner = rhs_spec.shape[rhs_axes.inner_idx];

    if (!IsPositiveIfStatic(lhs_inner) || !IsPositiveIfStatic(rhs_inner)) {
        return Status::InvalidArgument("MatMul inner dimensions must be positive when static");
    }

    // Static inner-dimension mismatch is a hard error; non-static equality
    // is deferred to runtime as DimEqualConstraint in InferOutputShapes.
    if (lhs_inner.IsStatic() && rhs_inner.IsStatic() && lhs_inner != rhs_inner) {
        return Status::InvalidArgument(
                "MatMul lhs/rhs inner dimensions are statically incompatible");
    }

    // Batch axes must be broadcastable.
    const SymbolicShape lhs_batch = MakeBatchShape(lhs_spec.shape, lhs_rank);
    const SymbolicShape rhs_batch = MakeBatchShape(rhs_spec.shape, rhs_rank);
    const auto broadcast_result = InferBroadcastShape(lhs_batch, rhs_batch);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    return Status::Ok();
}

StatusOr<InferenceResult> MatMulOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "MatMul expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }

    const auto& lhs_spec = inputs[0];
    const auto& rhs_spec = inputs[1];

    if (lhs_spec.dtype != DataType::Float32() || rhs_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("MatMul only supports float32 inputs in Phase 1");
    }

    const auto lhs_rank_opt = lhs_spec.shape.rank();
    if (!lhs_rank_opt.has_value() || *lhs_rank_opt < 2) {
        return Status::InvalidArgument("MatMul lhs shape must have rank >= 2");
    }

    const auto rhs_rank_opt = rhs_spec.shape.rank();
    if (!rhs_rank_opt.has_value() || *rhs_rank_opt < 2) {
        return Status::InvalidArgument("MatMul rhs shape must have rank >= 2");
    }

    const size_t lhs_rank = *lhs_rank_opt;
    const size_t rhs_rank = *rhs_rank_opt;
    const auto rhs_axes = ResolveRhsAxes(rhs_rank, params_.transpose_rhs);

    const ShapeSymbol& lhs_inner = lhs_spec.shape[lhs_rank - 1];
    const ShapeSymbol& rhs_inner = rhs_spec.shape[rhs_axes.inner_idx];
    const ShapeSymbol& lhs_outer = lhs_spec.shape[lhs_rank - 2];
    const ShapeSymbol& rhs_outer = rhs_spec.shape[rhs_axes.outer_idx];

    if (!IsPositiveIfStatic(lhs_inner) || !IsPositiveIfStatic(rhs_inner)) {
        return Status::InvalidArgument("MatMul inner dimensions must be positive when static");
    }
    if (!IsPositiveIfStatic(lhs_outer) || !IsPositiveIfStatic(rhs_outer)) {
        return Status::InvalidArgument("MatMul outer dimensions must be positive when static");
    }

    if (lhs_inner.IsStatic() && rhs_inner.IsStatic() && lhs_inner != rhs_inner) {
        return Status::InvalidArgument(
                "MatMul lhs/rhs inner dimensions are statically incompatible");
    }

    // Broadcast batch axes (lhs.shape[:-2] vs rhs.shape[:-2]).
    const SymbolicShape lhs_batch = MakeBatchShape(lhs_spec.shape, lhs_rank);
    const SymbolicShape rhs_batch = MakeBatchShape(rhs_spec.shape, rhs_rank);
    const auto broadcast_result = InferBroadcastShape(lhs_batch, rhs_batch);
    if (!broadcast_result.ok()) {
        return broadcast_result.status();
    }

    // Output shape = broadcast_batch + [M, N] where
    //   M = lhs.shape[-2], N = effective_rhs.shape[-1] (i.e. rhs_outer).
    const auto& batch_dims = broadcast_result->output_shape.shape();
    std::vector<ShapeSymbol> output_dims;
    if (batch_dims.has_value()) {
        output_dims.reserve(batch_dims->size() + 2);
        for (const auto& dim: *batch_dims) {
            output_dims.push_back(dim);
        }
    }
    output_dims.push_back(lhs_outer);
    output_dims.push_back(rhs_outer);

    TensorSpec output_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape(std::move(output_dims)),
    };

    std::vector<ShapeConstraint> runtime_checks;

    // Emit DimEqualConstraint when inner dimensions cannot be proven equal
    // statically. Both symbolic-unknown and static-vs-symbolic pairs land
    // here; only identical statics or matching symbols are proven.
    if (lhs_inner != rhs_inner) {
        runtime_checks.push_back(ShapeConstraint{
                .condition = DimEqualConstraint{
                        .lhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput,
                                                          .tensor_idx = 0},
                                .dim_index = lhs_rank - 1,
                        },
                        .rhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = rhs_axes.inner_idx,
                        },
                },
                .error_context = "MatMul lhs inner dimension must equal rhs contraction dimension",
        });
    }

    // DeferredBroadcastAxis indices reference the batch shapes passed to
    // InferBroadcastShape, which align 1:1 with the leading axes of the full
    // lhs/rhs shapes (batch = shape[:-2]). No remapping is needed.
    for (const auto& deferred: broadcast_result->deferred_axes) {
        runtime_checks.push_back(ShapeConstraint{
                .condition = DimBroadcastableConstraint{
                        .lhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput,
                                                          .tensor_idx = 0},
                                .dim_index = deferred.lhs_axis,
                        },
                        .rhs = DimLocator{
                                .tensor_port = TensorPort{.direction = TensorPortType::kInput, .tensor_idx = 1},
                                .dim_index = deferred.rhs_axis,
                        },
                },
                .error_context = "MatMul batch dimensions are not broadcastable",
        });
    }

    return InferenceResult{
            .outputs = {std::move(output_spec)},
            .runtime_checks = std::move(runtime_checks),
    };
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

}// namespace aethermind
