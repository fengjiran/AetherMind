#ifndef AETHERMIND_OPERATORS_OPS_KVCACHE_UPDATE_OP_H
#define AETHERMIND_OPERATORS_OPS_KVCACHE_UPDATE_OP_H

/// @file kvcache_update_op.h
/// @brief Shared dtype contract for KV-cache update inference.

#include "aethermind/dtypes/data_type.h"

#include <algorithm>

namespace aethermind {

/// @brief Supported dtype set for the KVCacheUpdate operator.
///
/// Two tuple families are legal: all four tensor inputs (k, v, k_cache_in,
/// v_cache_in) share one dtype from this set, or the P3 mixed KV16 boundary
/// k/v activation=F32 with k_cache_in/v_cache_in in the same F16 or BF16.
/// Activations must agree and caches must agree; no other mixed combination
/// declares an implicit conversion. Cache-out follows cache-in verbatim.
inline const std::array kKVCacheUpdateSupportedDTypes = {
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

} // namespace aethermind

#endif
