#ifndef AETHERMIND_OPERATORS_ROPE_OP_H
#define AETHERMIND_OPERATORS_ROPE_OP_H

/// @file rope_op.h
/// @brief Shared dtype contract for rotary-position-embedding inference.

#include "aethermind/dtypes/data_type.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {

/// @brief Supported dtype set for the RoPE operator.
///
/// q and k must share the same dtype from this set. Mixed dtypes between q
/// and k are rejected: no implicit conversion policy is declared by the
/// operator params. Output dtype follows q (and k respectively).
inline const std::array<DataType, 3> kRoPESupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether a dtype is supported by the RoPE operator.
///
/// @param dtype The data type to check.
/// @return True if `dtype` is in kRoPESupportedDTypes.
/// @note Backend kernel dispatch must reference the same set when adding new
///       dtype paths.
inline bool IsRoPESupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kRoPESupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent "unsupported dtype" error message for RoPE.
///
/// @param context Caller name (e.g. "RoPE q", "RoPE k") prepended to the
///                fixed list of supported dtypes.
/// @return Error message string with the caller context and supported dtype list.
inline std::string MakeRoPEUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
