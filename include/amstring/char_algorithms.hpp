// char_traits.hpp - Character traits helpers for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_CHAR_TRAITS_HPP
#define AETHERMIND_AMSTRING_CHAR_TRAITS_HPP

#include <cstddef>
#include <cstring>
#include <string>

namespace aethermind {

// Char algorithms using std::char_traits
// Encapsulates common operations for any CharT
template<typename CharT, typename Traits = std::char_traits<CharT>>
struct CharAlgorithm {
    using char_type = CharT;
    using traits_type = Traits;
    using size_type = std::size_t;

    static constexpr CharT null_char() noexcept {
        return CharT{};
    }

    static size_type length(const CharT* s) noexcept {
        return Traits::length(s);
    }

    static void copy(CharT* dst, const CharT* src, size_type n) noexcept {
        Traits::copy(dst, src, n);
    }

    static void move(CharT* dst, const CharT* src, size_type n) noexcept {
        Traits::move(dst, src, n);
    }

    static CharT* find(CharT* s, size_type n, CharT ch) noexcept {
        return Traits::find(s, n, ch);
    }

    static const CharT* find(const CharT* s, size_type n, CharT ch) noexcept {
        return Traits::find(s, n, ch);
    }

    static int compare(const CharT* s1, const CharT* s2, size_type n) noexcept {
        return Traits::compare(s1, s2, n);
    }

    static void assign(CharT* dst, size_type n, CharT ch) noexcept {
        Traits::assign(dst, n, ch);
    }

    static void assign(CharT& dst, CharT ch) noexcept {
        Traits::assign(dst, ch);
    }

    static bool eq(CharT a, CharT b) noexcept {
        return Traits::eq(a, b);
    }

    static bool lt(CharT a, CharT b) noexcept {
        return Traits::lt(a, b);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CHAR_TRAITS_HPP