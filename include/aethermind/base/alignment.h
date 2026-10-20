#ifndef AETHERMIND_BASE_ALIGNMENT_H
#define AETHERMIND_BASE_ALIGNMENT_H

/// @file alignment.h
/// @brief Alignment validation and checked offset alignment shared by memory
/// and execution resources.

#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"

#include <bit>
#include <cstddef>

namespace aethermind {

/// @brief Checks whether an alignment is a non-zero power of two.
/// @param alignment Alignment in bytes.
/// @return True for a non-zero power of two, including byte alignment.
AM_NODISCARD constexpr bool IsValidAlignment(size_t alignment) noexcept {
    return std::has_single_bit(alignment);
}

/// @brief Aligns an offset up to the specified alignment boundary.
/// @param offset Current offset.
/// @param alignment Target alignment (non-zero power of two).
/// @return The smallest aligned offset not less than `offset`, kInvalidArgument
///         for invalid alignment, or kOverflow if the result is not representable
///         in size_t. Diagnostic allocation failure returns kResourceExhausted
///         with an empty message. Successful calls allocate no memory.
/// @post On success: result >= offset and (result % alignment) == 0.
StatusOr<size_t> AlignOffsetUp(size_t offset, size_t alignment) noexcept;

} // namespace aethermind

#endif // AETHERMIND_BASE_ALIGNMENT_H
