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

// Debug mode invariant checking
// Enable in debug builds, disable in release
#ifdef NDEBUG
constexpr bool kEnableInvariantCheck = false;
#else
constexpr bool kEnableInvariantCheck = true;
#endif

// SSO threshold for stable_layout_policy
// Computed as: sizeof(heap_rep) / sizeof(CharT) - 1
// This is a template-based calculation, not a fixed constant

// Initial heap capacity floor
// When transitioning from small to heap, minimum allocation
constexpr size_t kMinHeapCapacity = 32;

// Capacity growth factor
// new_cap = max(required, old_cap + old_cap / 2)
constexpr size_t kGrowthFactorDenominator = 2;

}// namespace config
}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CONFIG_HPP