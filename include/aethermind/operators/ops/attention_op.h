#ifndef AETHERMIND_OPERATORS_OPS_ATTENTION_OP_H
#define AETHERMIND_OPERATORS_OPS_ATTENTION_OP_H

/// @file attention_op.h
/// @brief Shared dtype contract for Attention semantic inference.

#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief Supported dtype set for the Attention operator.
///
/// q, K cache, and V cache must share the same dtype from this set. Mixed
/// dtypes between q and the caches are rejected: no implicit conversion
/// policy is declared by the operator params. Output dtype follows q.
inline const std::array<DataType, 3> kAttentionSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether a dtype is supported by the Attention operator.
///
/// @param dtype The data type to check.
/// @return True if `dtype` is in kAttentionSupportedDTypes.
/// @note Backend kernel dispatch must reference the same set when adding new
///       dtype paths.
inline bool IsAttentionSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kAttentionSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent "unsupported dtype" error message for Attention.
///
/// @param context Caller name (e.g. "Attention q", "CpuAttentionKernel k_cache")
///                prepended to the fixed list of supported dtypes.
/// @return Error message string with the caller context and supported dtype list.
inline std::string MakeAttentionUnsupportedDTypeMessage(
        std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

} // namespace aethermind

#endif
