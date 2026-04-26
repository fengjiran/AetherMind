// allocator_support.hpp - Allocator support utilities for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_ALLOCATOR_SUPPORT_HPP
#define AETHERMIND_AMSTRING_ALLOCATOR_SUPPORT_HPP

#include <memory>

namespace aethermind {

// Allocator traits helper
// Simplifies allocator operations and propagation
template<typename Allocator>
struct AllocatorTraitsHelper {
    using alloc_type = Allocator;
    using traits = std::allocator_traits<alloc_type>;

    using value_type = traits::value_type;
    using pointer = traits::pointer;
    using const_pointer = traits::const_pointer;
    using size_type = traits::size_type;
    using difference_type = traits::difference_type;

    static constexpr bool propagate_on_copy_assignment = traits::propagate_on_container_copy_assignment::value;

    static constexpr bool propagate_on_move_assignment = traits::propagate_on_container_move_assignment::value;

    static constexpr bool propagate_on_swap = traits::propagate_on_container_swap::value;

    static constexpr bool is_always_equal = traits::is_always_equal::value;

    static alloc_type select_on_copy(const alloc_type& a) {
        return traits::select_on_container_copy_construction(a);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_ALLOCATOR_SUPPORT_HPP
