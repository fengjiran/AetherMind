// string_fwd.hpp - Forward declarations for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_STRING_FWD_HPP
#define AETHERMIND_AMSTRING_STRING_FWD_HPP

#include <memory>
#include <string>

namespace aethermind {

// Forward declarations for basic_string
template <
    typename CharT,
    typename Traits = std::char_traits<CharT>,
    typename Allocator = std::allocator<CharT>
>
class basic_string;

// Type aliases
using string    = basic_string<char>;
using u8string  = basic_string<char8_t>;
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;
using wstring   = basic_string<wchar_t>;

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STRING_FWD_HPP