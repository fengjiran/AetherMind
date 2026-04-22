#ifndef AMSTRING_CATEGORY_HPP
#define AMSTRING_CATEGORY_HPP

#include <cstdint>

namespace aethermind {

// Storage category for Small String Optimization (SSO)
// First version: Small + Heap (Medium/Large both heap exclusive)
enum class Category : uint8_t {
    Small = 0,
    Medium = 1,// Heap exclusive ownership
    Large = 2, // Reserved, no special strategy in first version
};

// Category encoding strategy (first version):
// - Small: last CharT stores encoded metadata (kSmallCapacity - size)
// - Medium/Large: capacity_with_category stores category in bits
//
// This is simpler than fbstring byte-level trick, easier to support
// multiple CharT types (char, char8_t, char16_t, char32_t, wchar_t)
constexpr uint8_t kCategoryMask = 0x03;// Last 2 bits for category

}// namespace aethermind

#endif// AMSTRING_CATEGORY_HPP