#ifndef AETHERMIND_OPERATORS_OPS_EMBEDDING_OP_H
#define AETHERMIND_OPERATORS_OPS_EMBEDDING_OP_H

/// @file embedding_op.h
/// @brief Embedding lookup semantic contract.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

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

} // namespace aethermind

#endif
