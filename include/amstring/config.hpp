// config.hpp - Configuration constants for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_CONFIG_HPP
#define AETHERMIND_AMSTRING_CONFIG_HPP

#include <bit>
#include <cstddef>

namespace aethermind {
namespace config {

// Endianness detection
// Used for compact_layout_policy<char> encoding
constexpr bool kIsLittleEndian = std::endian::native == std::endian::little;

// Initial heap capacity floor
// When transitioning from small to heap, minimum allocation
constexpr size_t kMinHeapCapacity = 32;

// Capacity growth factor
// new_cap = max(required, old_cap + old_cap / 2)
constexpr size_t kGrowthFactorDenominator = 2;

}// namespace config
}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CONFIG_HPP
