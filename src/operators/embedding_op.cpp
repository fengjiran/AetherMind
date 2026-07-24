#include "aethermind/operators/embedding_op.h"
#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"

namespace aethermind {
Status EmbeddingOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument(
                "Embedding Prepare requires OperatorContext.backend");
    }

    const auto resolved = ctx.backend->ResolveKernelInfo(
            OpType::kEmbedding,
            ctx.selector);
    if (!resolved.ok()) {
        return resolved.status();
    }

    resolved_kernel_ = resolved.value();
    if (resolved_kernel_.fn == nullptr) {
        return Status::Internal("Embedding Prepare resolved a kernel with null fn");
    }
    return Status::Ok();
}

Status EmbeddingOp::Run(KernelContext& ctx,
                        const RuntimeBindingContext& bindings,
                        size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status::FailedPrecondition("Embedding Run called before Prepare");
    }

    const auto binding = bindings.GetStepTensorBinding(step_index);
    if (!binding.ok()) {
        return binding.status();
    }

    const auto* b = binding.value();
    if (b->inputs.size() != 2) {
        return Status::InvalidArgument(
                "Embedding requires 2 input tensor bindings, got " +
                std::to_string(b->inputs.size()));
    }

    if (b->outputs.size() != 1) {
        return Status::InvalidArgument(
                "Embedding requires 1 output tensor binding, got " +
                std::to_string(b->outputs.size()));
    }

    // Phase 1 CPU-first: construct CPU-specific params directly. Phase 2 should
    // inject params construction via Backend to support multiple backends.
    return InvokeResolvedKernel(ctx, b->inputs, b->outputs);
}

AM_REGISTER_OPERATOR(OpType::kEmbedding, EmbeddingOp)


namespace detail {

StatusOr<InferenceResult> InferEmbedding(const OpParams& params,
                                         std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<EmbeddingParams>(params)) {
        return Status::InvalidArgument("Embedding node requires EmbeddingParams");
    }

    if (inputs.size() != 2) {
        return Status::InvalidArgument("Embedding requires exactly 2 inputs");
    }

    const TensorSpec& input_ids_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    const auto& input_ids_shape = input_ids_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    const auto input_rank = input_ids_shape.rank();
    if (!input_rank.has_value() || *input_rank < 1) {
        return Status::InvalidArgument("Embedding input must have rank >= 1");
    }

    // Batch dimensions must be positive when statically known.
    for (const auto dim: input_ids_shape) {
        if (!IsPositiveIfStatic(dim)) {
            return Status::InvalidArgument(
                    "Embedding token_ids dimension must be positive when statically known.");
        }
    }

    if (!HasRank(weight_shape, 2)) {
        return Status::InvalidArgument("Embedding weight must be rank 2");
    }

    for (const auto dim: weight_shape) {
        if (!IsPositiveIfStatic(dim)) {
            return Status::InvalidArgument(
                    "Embedding weight dimension must be positive when statically known.");
        }
    }

    if (!IsSupportedTokenIdDType(input_ids_spec.dtype)) {
        return Status::InvalidArgument(
                "Embedding token_ids must be int32, int64, or uint32");
    }

    if (!IsEmbeddingSupportedWeightDType(weight_spec.dtype)) {
        return Status::InvalidArgument(
                MakeEmbeddingUnsupportedWeightDTypeMessage("Embedding"));
    }


    InferenceResult result;
    result.outputs.push_back({weight_spec.dtype, SymbolicShape({input_ids_shape[0], weight_shape[1]})});
    return result;
}

}// namespace detail

}// namespace aethermind
