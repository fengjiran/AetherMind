/// \file
/// Implementation of memory debugging utilities.

#include "aethermind/utils/memory_utils.h"

#include <cstring>

namespace aethermind {

void FillMemoryJunk(void* data, size_t nbytes) {
    if (data == nullptr || nbytes == 0) {
        return;
    }

    constexpr int64_t kJunkPattern = GetJunkPattern64();

    // Fill in 64-bit chunks for efficiency
    const size_t int64_count = nbytes / sizeof(int64_t);
    const size_t remaining_bytes = nbytes % sizeof(int64_t);

    auto* data_i64 = static_cast<int64_t*>(data);
    for (size_t i = 0; i < int64_count; ++i) {
        data_i64[i] = kJunkPattern;
    }

    // Fill remaining bytes
    if (remaining_bytes > 0) {
        memcpy(data_i64 + int64_count, &kJunkPattern, remaining_bytes);
    }
}

}// namespace aethermind
