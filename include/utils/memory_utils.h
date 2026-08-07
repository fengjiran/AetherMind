/// \file
/// Memory debugging utilities.
///
/// Provides functions for detecting uninitialized memory access patterns
/// by filling memory with distinctive junk values.

#ifndef AETHERMIND_UTILS_MEMORY_UTILS_H
#define AETHERMIND_UTILS_MEMORY_UTILS_H

#include <cstddef>
#include <cstdint>

namespace aethermind {

/// Fills a memory region with a junk pattern for debugging uninitialized access.
///
/// The pattern is chosen to be:
/// - NaN when interpreted as float32 or float64
/// - Large integer values that cause visible bugs when used
///
/// This helps catch bugs where code reads from uninitialized memory,
/// as the junk values are unlikely to be valid data.
///
/// \param data Pointer to the memory region to fill. Must not be null.
/// \param nbytes Number of bytes to fill. Must be > 0.
///
/// \note This function is intended for debugging purposes only.
///       Production code should not rely on junk-filling behavior.
void FillMemoryJunk(void* data, size_t nbytes);

/// Returns the 32-bit junk pattern value.
/// \return The pattern used by FillMemoryJunk for testing.
constexpr int32_t GetJunkPattern32() noexcept {
    return 0x7fedbeef;
}

/// Returns the 64-bit junk pattern value.
/// \return The pattern used by FillMemoryJunk for testing.
constexpr int64_t GetJunkPattern64() noexcept {
    // 32-bit pattern repeated twice
    return static_cast<int64_t>(GetJunkPattern32()) << 32 | GetJunkPattern32();
}

}// namespace aethermind

#endif// AETHERMIND_UTILS_MEMORY_UTILS_H