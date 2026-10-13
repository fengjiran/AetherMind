#ifndef AETHERMIND_KV_CACHE_KV_CACHE_COMMIT_ACCESS_H
#define AETHERMIND_KV_CACHE_KV_CACHE_COMMIT_ACCESS_H

/// @file kv_cache_commit_access.h
/// @brief Private execution capability for committing a successful KV append.

#include "aethermind/kv_cache/kv_cache_session_view.h"

namespace aethermind::detail {

/// This capability is intentionally private to the implementation tree. It
/// must be called immediately after a complete execution transaction succeeds;
/// callers must not expose it as a general session mutation API. Internal test
/// fixtures may include this private header to exercise the same capability.
class KVCacheCommitAccess {
public:
    static Status CommitAppend(KVCacheSessionView& view,
                               size_t begin,
                               size_t end) noexcept {
        return view.CommitAppend(begin, end);
    }
};

} // namespace aethermind::detail

#endif // AETHERMIND_KV_CACHE_KV_CACHE_COMMIT_ACCESS_H
