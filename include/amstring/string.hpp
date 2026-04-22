// string.hpp - Convenience header for amstring type aliases
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_STRING_HPP
#define AETHERMIND_AMSTRING_STRING_HPP

#include "basic_string.hpp"
#include "string_fwd.hpp"

namespace aethermind {

// Type aliases at namespace level for convenience
// These are the primary user-facing types

using string    = basic_string<char>;
using u8string  = basic_string<char8_t>;
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;
using wstring   = basic_string<wchar_t>;

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STRING_HPP