#ifndef AETHERMIND_OPERATORS_OPS_EMBEDDING_OP_H
#define AETHERMIND_OPERATORS_OPS_EMBEDDING_OP_H

/// @file embedding_op.h
/// @brief Embedding lookup semantics and executable operator declaration.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

#include <array>
#include <string>

namespace aethermind {

/// @brief Integer dtypes accepted for Embedding token IDs.
///
/// All Embedding token-ID validation sites must reference this set instead of
/// maintaining private copies.
inline const std::array<DataType, 3> kEmbeddingSupportedTokenIdDTypes = {
        DataType::Int(32),
        DataType::Int(64),
        DataType::UInt(32),
};

/// @brief Checks whether Embedding accepts a token-ID dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kEmbeddingSupportedTokenIdDTypes`.
inline bool IsSupportedTokenIdDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kEmbeddingSupportedTokenIdDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Floating-point dtypes accepted for Embedding weights.
///
/// Output dtype follows the weight dtype.
inline const std::array<DataType, 3> kEmbeddingSupportedWeightDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether Embedding accepts a weight dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kEmbeddingSupportedWeightDTypes`.
inline bool IsEmbeddingSupportedWeightDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kEmbeddingSupportedWeightDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-weight-dtype message for Embedding.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted weight dtypes.
inline std::string MakeEmbeddingUnsupportedWeightDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " weight only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

/// @brief Embedding lookup operator whose output dtype follows the weight dtype.
class EmbeddingOp final : public Operator {
public:
    using Params = EmbeddingParams;

    explicit EmbeddingOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kEmbedding;
    }

    Status Prepare(OperatorContext& ctx) override;

    Status Run(KernelContext& ctx,
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
