#include "aethermind/base/alignment.h"
#include "utils/overflow_check.h"

#include <new>

namespace aethermind {

StatusOr<size_t> AlignOffsetUp(size_t offset, size_t alignment) noexcept try {
    if (!IsValidAlignment(alignment)) {
        return Status::InvalidArgument(
                "Alignment must be a non-zero power of two");
    }

    size_t rounded_offset = 0;
    if (CheckOverflowAdd(offset, alignment - 1, &rounded_offset)) {
        return Status::Overflow("Offset alignment overflowed size_t");
    }

    return rounded_offset & ~(alignment - 1);
} catch (const std::bad_alloc&) {
    // Error diagnostics may allocate; reporting OOM must not allocate again.
    return Status::ResourceExhausted({});
}

} // namespace aethermind
