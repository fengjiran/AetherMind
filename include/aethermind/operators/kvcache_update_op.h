#ifndef AETHERMIND_OPERATORS_KVCACHE_UPDATE_OP_H
#define AETHERMIND_OPERATORS_KVCACHE_UPDATE_OP_H

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

#include "aethermind/dtypes/data_type.h"

namespace aethermind {

/// Single source of truth for the dtype set supported by the KVCacheUpdate
/// operator. All four tensor inputs (k, v, k_cache_in, v_cache_in) must share
/// the same dtype from this set. Mixed dtypes between activations and cache
/// are rejected: an implicit conversion strategy is not declared by the
/// operator params and no kernel contract exists for it. If mixed cache dtype
/// becomes a real requirement, add an explicit conversion policy instead of
/// silently accepting arbitrary combinations.
inline const std::array<DataType, 3> kKVCacheUpdateSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// Returns true if `dtype` is in `kKVCacheUpdateSupportedDTypes`. Used by
/// operator-level validation to keep the dtype check in one place. Backend
/// kernel dispatch must reference this same set when adding new dtype paths.
inline bool IsKVCacheUpdateSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kKVCacheUpdateSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported dtype" error message for KVCacheUpdate-
/// related validation points. `context` is the caller name (e.g. "KVCacheUpdate
/// k", "CpuKVCacheUpdateKernel v") prepended to a fixed list of supported
/// dtypes, so every validation site reports the same set.
inline std::string MakeKVCacheUpdateUnsupportedDTypeMessage(
        std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

}// namespace aethermind

#endif
