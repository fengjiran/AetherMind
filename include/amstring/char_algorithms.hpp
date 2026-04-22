#ifndef AMSTRING_CHAR_ALGORITHMS_HPP
#define AMSTRING_CHAR_ALGORITHMS_HPP

#include "macros.h"

#include <cstring>
#include <string>
#include <type_traits>

namespace aethermind {

// Character algorithm helpers
// First version: simple implementations using Traits

// Copy characters from src to dst
template<typename CharT, typename Traits>
void char_copy(CharT* dst, const CharT* src, size_t n) {
    if (n > 0) AM_LIKELY {
            Traits::copy(dst, src, n);
        }
}

// Move characters (handles overlapping ranges)
template<typename CharT, typename Traits>
void char_move(CharT* dst, const CharT* src, size_t n) {
    if (n > 0) AM_LIKELY {
            Traits::move(dst, src, n);
        }
}

// Fill characters
template<typename CharT, typename Traits>
void char_fill(CharT* dst, size_t n, CharT ch) {
    if (n > 0) AM_LIKELY {
            Traits::assign(dst, n, ch);
        }
}

// Compare characters
template<typename CharT, typename Traits>
int char_compare(const CharT* lhs, const CharT* rhs, size_t n) {
    return Traits::compare(lhs, rhs, n);
}

// Find character in range
template<typename CharT, typename Traits>
const CharT* char_find(const CharT* s, size_t n, CharT ch) {
    return Traits::find(s, n, ch);
}

// Length of null-terminated string
template<typename CharT, typename Traits>
size_t char_length(const CharT* s) {
    return Traits::length(s);
}

// Specialization for char using std::memcpy for efficiency
// (Only used in char-optimized core later, not in first version generic core)

}// namespace aethermind

#endif// AMSTRING_CHAR_ALGORITHMS_HPP