// string_fwd.hpp - Forward declarations for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_STRING_FWD_HPP
#define AETHERMIND_AMSTRING_STRING_FWD_HPP

#include <memory>
#include <string>

namespace aethermind {

// Forward declarations for BasicString
template <
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>
>
class BasicString;

// Type aliases
using string    = BasicString<char>;
using u8string  = BasicString<char8_t>;
using u16string = BasicString<char16_t>;
using u32string = BasicString<char32_t>;
using wstring   = BasicString<wchar_t>;

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STRING_FWD_HPP
