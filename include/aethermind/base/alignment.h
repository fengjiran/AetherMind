#ifndef AETHERMIND_BASE_ALIGNMENT_H
#define AETHERMIND_BASE_ALIGNMENT_H

/// @file alignment.h
/// @brief Alignment validation shared by memory and execution resources.

#include "aethermind/base/macros.h"

#include <bit>
#include <cstddef>

namespace aethermind {

/// @brief Checks whether an alignment is a non-zero power of two.
/// @param alignment Alignment in bytes.
/// @return True for a non-zero power of two, including byte alignment.
AM_NODISCARD constexpr bool IsValidAlignment(size_t alignment) noexcept {
    return std::has_single_bit(alignment);
}

} // namespace aethermind

#endif // AETHERMIND_BASE_ALIGNMENT_H
