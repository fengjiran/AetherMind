#ifndef AMSTRING_CONFIG_HPP
#define AMSTRING_CONFIG_HPP

#include <bit>

// amstring configuration macros
// First version: simple configuration, no advanced optimizations

namespace aethermind {

namespace config {

// First version design decisions (frozen):
// - Object size: target 24 bytes
// - Small capacity: sizeof(MediumLarge)/sizeof(CharT) - 1
// - Category encoding: Small uses last CharT for metadata (no byte trick)
// - Large: reserved category, no special strategy
// - COW: disabled
// - safe over-read: disabled (first version)
// - SIMD: disabled

// Optimization flags (all disabled for first version)
#define AMSTRING_ENABLE_SAFE_OVER_READ 0
#define AMSTRING_ENABLE_BRANCHLESS_SIZE 0
#define AMSTRING_ENABLE_SIMD_FIND 0
#define AMSTRING_ENABLE_COW 0

// Debug invariant checking
#ifdef NDEBUG
#define AMSTRING_CHECK_INVARIANTS 0
#else
#define AMSTRING_CHECK_INVARIANTS 1
#endif

constexpr static auto kIsLittleEndian = std::endian::native == std::endian::little;

}// namespace config
}// namespace aethermind

#endif// AMSTRING_CONFIG_HPP