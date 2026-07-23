#include "aethermind/operators/embedding_op.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_registry.h"

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include <string>

namespace aethermind {
Status EmbeddingOp::Prepare(OperatorContext& ctx) {
    if (ctx.backend == nullptr) {
        return Status::InvalidArgument("Embedding Prepare requires OperatorContext.backend");
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

namespace {

bool IsSupportedTokenIdDType(const DataType& dtype) {
    return dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::UInt(32);
}

}// namespace

StatusOr<InferenceResult> InferEmbedding(const OpParams& params,
                                         std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<EmbeddingParams>(params)) {
        return Status::InvalidArgument("Embedding node requires EmbeddingParams");
    }
    if (inputs.size() != 2) {
        return Status::InvalidArgument("Embedding requires exactly 2 inputs");
    }
    const TensorSpec& token_spec = inputs[0];
    const TensorSpec& weight_spec = inputs[1];
    if (!IsSupportedTokenIdDType(token_spec.dtype)) {
        return Status::InvalidArgument("Embedding token_ids must be int32, int64, or uint32");
    }
    if (weight_spec.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Embedding weight must be float32");
    }
    if (!HasRank(token_spec.shape, 1)) {
        return Status::InvalidArgument("Embedding token_ids must be rank 1");
    }
    if (!HasRank(weight_spec.shape, 2)) {
        return Status::InvalidArgument("Embedding weight must be rank 2");
    }
    const auto& tokens_shape = token_spec.shape;
    const auto& weight_shape = weight_spec.shape;
    if (tokens_shape[0].IsStatic() && tokens_shape[0].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Embedding token_ids dimension must be positive");
    }
    if (weight_shape[0].IsStatic() && weight_shape[0].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Embedding weight dimension 0 must be positive");
    }
    if (weight_shape[1].IsStatic() && weight_shape[1].GetStaticValue() <= 0) {
        return Status::InvalidArgument("Embedding weight dimension 1 must be positive");
    }
    InferenceResult result;
    result.outputs.push_back({weight_spec.dtype, SymbolicShape({tokens_shape[0], weight_shape[1]})});
    return result;
}

}// namespace detail

}// namespace aethermind
