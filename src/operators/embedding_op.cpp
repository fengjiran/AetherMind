#include "aethermind/operators/embedding_op.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/kernels/cpu_embedding_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_registry.h"

#include <string>

namespace aethermind {
namespace {

bool IsSupportedTokenIdDType(const DataType& dtype) noexcept {
    return dtype == DataType::Int(32) ||
           dtype == DataType::Int(64) ||
           dtype == DataType::UInt(32);
}

}// namespace

Status EmbeddingOp::ValidateParams() const {
    return Status::Ok();
}

Status EmbeddingOp::CheckInputSpecs(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Embedding expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& token_ids = inputs[0];
    const auto& weight = inputs[1];

    if (!IsSupportedTokenIdDType(token_ids.dtype)) {
        return Status::InvalidArgument("Embedding token ids must be int32, int64, or uint32");
    }

    if (weight.dtype != DataType::Float32()) {
        return Status::InvalidArgument("Embedding only supports float32 weights in Phase 1");
    }

    if (!HasRank(token_ids.shape, 1)) {
        return Status::InvalidArgument("Embedding token ids must be rank-1");
    }

    if (!HasRank(weight.shape, 2)) {
        return Status::InvalidArgument("Embedding weight must be rank-2");
    }

    if (!IsPositiveIfStatic(token_ids.shape[0]) ||
        !IsPositiveIfStatic(weight.shape[0]) ||
        !IsPositiveIfStatic(weight.shape[1])) {
        return Status::InvalidArgument("Embedding token, vocab, and hidden sizes must be positive");
    }
    return Status::Ok();
}

StatusOr<InferenceResult> EmbeddingOp::InferOutputShapes(std::span<const TensorSpec> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Embedding expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }
    const auto& token_ids = inputs[0];
    const auto& weight = inputs[1];

    if (!HasRank(token_ids.shape, 1)) {
        return Status::InvalidArgument("Embedding token id shape must be rank-1");
    }

    if (!HasRank(weight.shape, 2)) {
        return Status::InvalidArgument("Embedding weight shape must be rank-2");
    }

    if (!IsPositiveIfStatic(token_ids.shape[0]) ||
        !IsPositiveIfStatic(weight.shape[0]) ||
        !IsPositiveIfStatic(weight.shape[1])) {
        return Status::InvalidArgument("Embedding token, vocab, and hidden sizes must be positive");
    }

    return InferenceResult{
            .outputs = {TensorSpec{
                    .dtype = weight.dtype,
                    .shape = SymbolicShape(std::vector<ShapeSymbol>{token_ids.shape[0], weight.shape[1]}),
            }},
            .runtime_checks = {},
    };
}

Status EmbeddingOp::Prepare(OperatorContext& ctx) {
    AM_RETURN_IF_ERROR(ValidateParams());
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
    return Status::Ok();
}

Status EmbeddingOp::Run(KernelContext& ctx,
                        const RuntimeBindingContext& bindings,
                        size_t step_index) const noexcept {
    if (resolved_kernel_.fn == nullptr) {
        return Status(StatusCode::kFailedPrecondition, "Embedding Run called before Prepare");
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

    CpuEmbeddingParams params{
            .token_ids_ = b->inputs[0],
            .weight_ = b->inputs[1],
            .output_ = b->outputs[0],
    };
    ctx.kernel_params = &params;
    return resolved_kernel_.fn(ctx);
}

AM_REGISTER_OPERATOR(OpType::kEmbedding, EmbeddingOp)

}// namespace aethermind
