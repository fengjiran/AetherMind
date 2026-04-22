#ifndef AMSTRING_BASIC_STRING_HPP
#define AMSTRING_BASIC_STRING_HPP

#include "core.hpp"
#include "string_fwd.hpp"

#include <memory>

namespace aethermind {

// Main string template - public API layer
// Wraps basic_string_core and provides std::basic_string-like interface
//
// First version (MVP):
// - Support char only
// - Basic constructors and accessors
// - Capacity management
// - Append operations
// - Find/compare operations
//
// Later milestones:
// - Full std::basic_string API coverage
// - Multi-CharT support
// - Allocator support
// - char-optimized core

template<
        typename CharT,
        typename Traits = std::char_traits<CharT>,
        typename Allocator = std::allocator<CharT>>
class basic_string {
public:
    using value_type = CharT;
    using traits_type = Traits;
    using allocator_type = Allocator;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = CharT&;
    using const_reference = const CharT&;
    using pointer = CharT*;
    using const_pointer = const CharT*;
    using iterator = CharT*;// TODO: proper iterator type
    using const_iterator = const CharT*;

    // TODO: Implement in Milestone 3
    // This is a skeleton for M0

private:
    basic_string_core<CharT, Traits, Allocator> core_;
};

// Type aliases (for convenience)
using string = basic_string<char>;
using u8string = basic_string<char8_t>;
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;
using wstring = basic_string<wchar_t>;

}// namespace aethermind

#endif// AMSTRING_BASIC_STRING_HPP