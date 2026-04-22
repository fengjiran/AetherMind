// layout_policy.hpp - LayoutPolicy interface definition for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_LAYOUT_POLICY_HPP

#include <cstddef>

namespace aethermind {

// LayoutPolicy interface contract
// All layout policies must implement these static methods
//
// Responsibilities:
// - Define storage_type (union of small array and heap representation)
// - Provide init_empty, init_small, init_heap
// - Provide is_small, is_heap category detection
// - Provide data, size, capacity extraction
// - Provide set_size, set_capacity modification
// - Provide destroy_heap cleanup
// - Provide check_invariants validation
//
// LayoutPolicy does NOT handle:
// - When to grow (that's GrowthPolicy)
// - Algorithm flow (that's basic_string_core)
// - Allocator calls (that's basic_string_core)

// Default layout policy selector
// Phase 1: All CharT use stable_layout_policy
// Phase 2: char switches to compact_layout_policy
template<typename CharT>
struct default_layout_policy;

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_LAYOUT_POLICY_HPP