#ifndef AETHERMIND_UTILS_MEMORY_UTILS_H
#define AETHERMIND_UTILS_MEMORY_UTILS_H

/// @file
/// @brief Utilities for filling memory with recognizable debug patterns.
///
/// Provides functions for detecting uninitialized memory access patterns
/// by filling memory with distinctive junk values.

#include <cstddef>
#include <cstdint>

namespace aethermind {

/// @brief Fills a memory region with a recognizable pattern.
///
/// The pattern is a NaN for floating-point interpretations and a large value
/// for integer interpretations, making accidental reads easier to diagnose.
///
/// @param data Destination memory. If `data` is null, the function does nothing;
///             otherwise it must point to at least `nbytes` writable bytes.
/// @param nbytes Number of bytes to fill. A zero length is a no-op.
/// @note This function is intended for debugging only. Production behavior
///       must not depend on the fill pattern.
void FillMemoryJunk(void* data, size_t nbytes);

/// @brief Returns the 32-bit pattern used by `FillMemoryJunk`.
/// @return The recognizable 32-bit debug pattern.
constexpr int32_t GetJunkPattern32() noexcept {
    return 0x7fedbeef;
}

/// @brief Returns the 64-bit pattern used by `FillMemoryJunk`.
/// @return The 32-bit pattern repeated in both halves of a 64-bit value.
constexpr int64_t GetJunkPattern64() noexcept {
    // Repeat the 32-bit pattern in both halves.
    return static_cast<int64_t>(GetJunkPattern32()) << 32 | GetJunkPattern32();
}

}// namespace aethermind

#endif// AETHERMIND_UTILS_MEMORY_UTILS_H
