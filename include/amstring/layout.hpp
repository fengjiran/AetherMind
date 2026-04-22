#ifndef AMSTRING_LAYOUT_HPP
#define AMSTRING_LAYOUT_HPP

#include "category.hpp"

#include <cstddef>

namespace aethermind {

// Storage layout for amstring
// Target: 24 bytes on 64-bit platforms
//
// Layout:
// - Small: CharT buffer inline, last CharT stores metadata
// - Heap: CharT* data, size_t size, size_t capacity_with_category
template<typename CharT>
struct MediumLarge {
    CharT* data_;
    size_t size_;
    size_t cap_;// capacity with category bits encoded
};

template<typename CharT>
union Storage {
    CharT small_[sizeof(MediumLarge<CharT>) / sizeof(CharT)];
    MediumLarge<CharT> ml_;
    std::byte bytes_[sizeof(MediumLarge<CharT>)];

    static constexpr size_t kStorageBytes = sizeof(MediumLarge<CharT>);
    static constexpr size_t kSmallArraySize = kStorageBytes / sizeof(CharT);
    static constexpr size_t kSmallCapacity = kSmallArraySize - 1;// Last CharT for metadata
};

// Compile-time checks
static_assert(sizeof(MediumLarge<char>) == 24, "MediumLarge<char> should be 24 bytes");
static_assert(Storage<char>::kSmallCapacity == 23, "Small capacity for char should be 23");

}// namespace aethermind

#endif// AMSTRING_LAYOUT_HPP