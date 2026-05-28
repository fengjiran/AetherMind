#include "aethermind/operators/embedding_op.h"

#include "aethermind/backend/backend.h"
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

Status EmbeddingOp::CheckShapes(std::span<const ShapeInfo> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Embedding expects exactly 2 inputs, got " + std::to_string(inputs.size()));
    }

    const auto& token_ids = inputs[0];
    const auto& weight = inputs[1];

    if (!IsSupportedTokenIdDType(token_ids.dtype_)) {
        return Status::InvalidArgument("Embedding token ids must be int32, int64, or uint32");
    }

    if (weight.dtype_ != DataType::Float32()) {
        return Status::InvalidArgument("Embedding only supports float32 weights in Phase 1");
    }

    if (token_ids.shape_.size() != 1) {
        return Status::InvalidArgument("Embedding token ids must be rank-1");
    }

    if (weight.shape_.size() != 2) {
        return Status::InvalidArgument("Embedding weight must be rank-2");
    }

    if (token_ids.shape_[0] <= 0 || weight.shape_[0] <= 0 || weight.shape_[1] <= 0) {
        return Status::InvalidArgument("Embedding token, vocab, and hidden sizes must be positive");
    }
    return Status::Ok();
}

StatusOr<std::vector<ShapeInfo>> EmbeddingOp::InferOutputShapes(std::span<const ShapeInfo> inputs) const {
    if (inputs.size() != 2) {
        return Status::InvalidArgument(
                "Embedding expects exactly 2 shape inputs, got " + std::to_string(inputs.size()));
    }
    const auto& token_ids = inputs[0];
    const auto& weight = inputs[1];

    if (token_ids.shape_.size() != 1) {
        return Status::InvalidArgument("Embedding token id shape must be rank-1");
    }

    if (weight.shape_.size() != 2) {
        return Status::InvalidArgument("Embedding weight shape must be rank-2");
    }

    const int64_t token_count = token_ids.shape_[0];
    const int64_t vocab_size = weight.shape_[0];
    const int64_t hidden_size = weight.shape_[1];
    if (token_count <= 0 || vocab_size <= 0 || hidden_size <= 0) {
        return Status::InvalidArgument("Embedding token, vocab, and hidden sizes must be positive");
    }

    return std::vector<ShapeInfo>{ShapeInfo{
            .dtype_ = weight.dtype_,
            .shape_ = {token_count, hidden_size},
    }};
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

AM_REGISTER_OPERATOR(OpType::kEmbedding, EmbeddingOp)

}// namespace aethermind
