#ifndef AETHERMIND_OPERATORS_EMBEDDING_OP_H
#define AETHERMIND_OPERATORS_EMBEDDING_OP_H

#include "aethermind/operators/operator.h"

namespace aethermind {

class EmbeddingOp final : public Operator {
public:
    struct Params {};

    explicit EmbeddingOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kEmbedding;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Embedding";
    }

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckShapes(std::span<const ShapeInfo> inputs) const override;
    AM_NODISCARD StatusOr<std::vector<ShapeInfo>> InferOutputShapes(
            std::span<const ShapeInfo> inputs) const override;

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;

    AM_NODISCARD Status Run(const KernelContext& ctx) const noexcept override {
        if (resolved_kernel_.fn == nullptr) {
            return Status(StatusCode::kFailedPrecondition, "Embedding Run called before Prepare");
        }
        return resolved_kernel_.fn(ctx);
    }

    AM_NODISCARD ResolvedKernel GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
