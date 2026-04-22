// invariant.hpp - Invariant checking utilities for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_INVARIANT_HPP
#define AETHERMIND_AMSTRING_INVARIANT_HPP

#include "config.hpp"

#include <cassert>
#include <cstddef>

namespace aethermind {

// Invariant check helper
// Only active in debug builds (kEnableInvariantCheck == true)
template<typename CharT>
inline void check_invariant_impl(
    const CharT* data,
    std::size_t size,
    std::size_t capacity
) noexcept {
    if (!config::kEnableInvariantCheck) {
        return;
    }

    // Core invariants:
    // 1. data() != nullptr (always valid, even for empty string)
    assert(data != nullptr);

    // 2. data()[size()] == CharT{} (null terminator)
    assert(data[size] == CharT{});

    // 3. size() <= capacity()
    assert(size <= capacity);
}

// Macro-style invariant check for development
// Can be disabled globally via NDEBUG
#define AM_STRING_INVARIANT_CHECK(expr) \
    do { if (aethermind::config::kEnableInvariantCheck) { assert(expr); } } while (0)

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_INVARIANT_HPP