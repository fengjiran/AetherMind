#ifndef AETHERMIND_OPERATORS_OPS_KVCACHE_UPDATE_OP_H
#define AETHERMIND_OPERATORS_OPS_KVCACHE_UPDATE_OP_H

/// @file kvcache_update_op.h
/// @brief Shared dtype contract for KV-cache update inference.

#include "aethermind/dtypes/data_type.h"

#include <algorithm>

namespace aethermind {

/// @brief Supported dtype set for the KVCacheUpdate operator.
///
/// All four tensor inputs (k, v, k_cache_in, v_cache_in) must share the same
/// dtype from this set. Mixed dtypes between activations and cache are
/// rejected: no implicit conversion policy is declared by the operator params.
/// If mixed cache dtype becomes a real requirement, add an explicit conversion
/// policy instead of silently accepting arbitrary combinations.
inline const std::array<DataType, 3> kKVCacheUpdateSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether a dtype is supported by the KVCacheUpdate operator.
///
/// @param dtype The data type to check.
/// @return True if `dtype` is in kKVCacheUpdateSupportedDTypes.
/// @note Backend kernel dispatch must reference the same set when adding new
///       dtype paths.
inline bool IsKVCacheUpdateSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kKVCacheUpdateSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent "unsupported dtype" error message for KVCacheUpdate.
///
/// @param context Caller name (e.g. "KVCacheUpdate k", "CpuKVCacheUpdateKernel v")
///                prepended to the fixed list of supported dtypes.
/// @return Error message string with the caller context and supported dtype list.
inline std::string MakeKVCacheUpdateUnsupportedDTypeMessage(
        std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
