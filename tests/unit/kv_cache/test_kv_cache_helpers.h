#ifndef AETHERMIND_TEST_KV_CACHE_HELPERS_H
#define AETHERMIND_TEST_KV_CACHE_HELPERS_H

#include "aethermind/kv_cache/kv_cache_session_view.h"

#include <cstring>

namespace aethermind::test {

// Metadata and pure-read fixtures must initialize every K/V row before making
// it visible. Real execution tests retain their kernel-written numerical data.
inline Status InitializeAndCommitKVForTest(KVCacheSessionView& view, size_t end) {
    if (!view.valid()) {
        return Status::FailedPrecondition("Test KV initialization requires a live view");
    }
    if (end < view.committed_tokens() || end > view.token_capacity()) {
        return Status::OutOfRange("Test KV initialization exceeds the reserved append range");
    }
    if (view.awaiting_prefill() && end < view.prompt_len()) {
        return Status::FailedPrecondition("Test KV initialization requires the complete Prefill");
    }

    while (view.committed_tokens() < end) {
        const size_t begin = view.committed_tokens();
        const size_t append_end = view.awaiting_prefill() ? view.prompt_len() : begin + 1;
        for (size_t layer = 0; layer < view.num_layers(); ++layer) {
            const auto append = view.BindLayerForAppend(layer, begin, append_end);
            if (!append.ok()) {
                return append.status();
            }
            const auto& storage = append->storage;
            const size_t row_bytes = storage.head_dim * static_cast<size_t>(storage.dtype.nbytes());
            for (size_t head = 0; head < storage.num_kv_heads; ++head) {
                for (size_t token = begin; token < append_end; ++token) {
                    const size_t offset = head * storage.head_stride_bytes + token * storage.token_stride_bytes;
                    std::memset(storage.key_data + offset, 0, row_bytes);
                    std::memset(storage.value_data + offset, 0, row_bytes);
                }
            }
        }
        const Status status = view.CommitAppend(begin, append_end);
        if (!status.ok()) {
            return status;
        }
    }
    return Status::Ok();
}

} // namespace aethermind::test

#endif // AETHERMIND_TEST_KV_CACHE_HELPERS_H
