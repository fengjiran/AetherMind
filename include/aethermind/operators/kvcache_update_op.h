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
/// operator's k/v activation inputs. All KVCacheUpdate-related validation
/// (semantic analysis in InferKVCacheUpdate, future CPU kernel dispatch) must
/// reference these definitions instead of maintaining private copies. The
/// Phase 1 CPU kernel currently implements only Float32; the semantic layer
/// accepts the full set below.
///
/// Note: only the k/v activation inputs (ports 0 and 1) are validated against
/// this set. The cache-in state ports (ports 2 and 3) are passed through to
/// the cache-out outputs verbatim, so their dtype is unconstrained at the
/// semantic layer and is determined by the runtime KV cache configuration.
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
