#ifndef AETHERMIND_MEMORY_MEMORY_JUNK_FILL_H
#define AETHERMIND_MEMORY_MEMORY_JUNK_FILL_H

#include <cstddef>

namespace aethermind::memory::detail {

/// @brief Fills allocator storage with a recognizable native-endian debug pattern.
/// @param data Destination with at least nbytes writable bytes; no alignment is
///             required. A null destination is a no-op.
/// @param nbytes Number of bytes to fill. Zero is a no-op.
/// @note Tensor values must not depend on the debug fill pattern.
void FillMemoryJunk(void* data, size_t nbytes) noexcept;

} // namespace aethermind::memory::detail

#endif // AETHERMIND_MEMORY_MEMORY_JUNK_FILL_H
