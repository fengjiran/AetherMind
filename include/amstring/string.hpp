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

using string    = BasicString<char>;
using u8string  = BasicString<char8_t>;
using u16string = BasicString<char16_t>;
using u32string = BasicString<char32_t>;
using wstring   = BasicString<wchar_t>;

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STRING_HPP
