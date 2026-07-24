#ifndef AETHERMIND_OPERATORS_EMBEDDING_OP_H
#define AETHERMIND_OPERATORS_EMBEDDING_OP_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

#include <array>
#include <string>

namespace aethermind {

/// Single source of truth for the dtype set supported by the Embedding
/// operator's token_ids input. All Embedding-related validation (semantic
/// analysis in InferEmbedding, future CPU kernel dispatch) must reference these
/// definitions instead of maintaining private copies.
inline const std::array<DataType, 3> kEmbeddingSupportedTokenIdDTypes = {
        DataType::Int(32),
        DataType::Int(64),
        DataType::UInt(32),
};

/// Returns true if `dtype` is a valid token_ids dtype (int32, int64, uint32).
/// Used by operator-level validation to keep the dtype check in one place.
inline bool IsSupportedTokenIdDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kEmbeddingSupportedTokenIdDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Single source of truth for the dtype set supported by the Embedding
/// operator's weight input. The semantic layer accepts these dtypes; the
/// Phase 1 CPU kernel currently implements only Float32. Output dtype
/// follows weight dtype.
inline const std::array<DataType, 3> kEmbeddingSupportedWeightDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// Returns true if `dtype` is a valid Embedding weight dtype
/// (float32, float16, bfloat16). Backend kernel dispatch must reference this
/// same set when adding new dtype paths.
inline bool IsEmbeddingSupportedWeightDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kEmbeddingSupportedWeightDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported weight dtype" error message for
/// Embedding-related validation points. `context` is the caller name
/// (e.g. "Embedding", "CpuEmbeddingKernel") prepended to a fixed list of
/// supported weight dtypes, so every validation site reports the same set.
inline std::string MakeEmbeddingUnsupportedWeightDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " weight only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

class EmbeddingOp final : public Operator {
public:
    using Params = EmbeddingParams;

    explicit EmbeddingOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kEmbedding;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Embedding";
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
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
