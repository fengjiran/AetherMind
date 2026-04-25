// invariant.hpp - Invariant checking utilities for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_INVARIANT_HPP
#define AETHERMIND_AMSTRING_INVARIANT_HPP

#include "config.hpp"
#include "utils/logging.h"

#include <cstddef>

namespace aethermind {

// Checks invariants that are common to every string data view.
// This helper deliberately knows nothing about layout state, probe bytes,
// capacity tags, allocator ownership, or Small/External classification.
template<typename CharT>
void CheckDataInvariants(const CharT* data,
                         std::size_t size,
                         std::size_t capacity) noexcept {
    AM_DCHECK(data != nullptr);
    AM_DCHECK(size <= capacity);
    AM_DCHECK(data[size] == CharT{});
}

// Macro-style invariant check for development
// Can be disabled globally via NDEBUG
#define AM_STRING_INVARIANT_CHECK(expr) \
    do {                                \
        AM_DCHECK(expr);                \
    } while (false)

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_INVARIANT_HPP
