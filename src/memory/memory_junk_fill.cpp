#include "memory_junk_fill.h"

#include <cstdint>
#include <cstring>

namespace aethermind::memory::detail {
namespace {

// Preserve the allocator's existing repeated binary32 NaN word representation.
constexpr uint64_t kJunkPattern = 0x7fedbeef7fedbeefULL;

} // namespace

void FillMemoryJunk(void* data, size_t nbytes) noexcept {
    if (data == nullptr || nbytes == 0) {
        return;
    }

    auto* bytes = static_cast<unsigned char*>(data);
    while (nbytes >= sizeof(kJunkPattern)) {
        std::memcpy(bytes, &kJunkPattern, sizeof(kJunkPattern));
        bytes += sizeof(kJunkPattern);
        nbytes -= sizeof(kJunkPattern);
    }
    if (nbytes != 0) {
        std::memcpy(bytes, &kJunkPattern, nbytes);
    }
}

} // namespace aethermind::memory::detail
