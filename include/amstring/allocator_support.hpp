// allocator_support.hpp - Allocator support utilities for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_ALLOCATOR_SUPPORT_HPP
#define AETHERMIND_AMSTRING_ALLOCATOR_SUPPORT_HPP

#include <memory>
#include <type_traits>

namespace aethermind {

// Allocator traits helper
// Simplifies allocator operations and propagation
template<typename Allocator>
struct allocator_traits_helper {
    using alloc_type = Allocator;
    using traits = std::allocator_traits<alloc_type>;

    using value_type = typename traits::value_type;
    using pointer = typename traits::pointer;
    using const_pointer = typename traits::const_pointer;
    using size_type = typename traits::size_type;
    using difference_type = typename traits::difference_type;

    static constexpr bool propagate_on_copy =
        traits::propagate_on_container_copy_assignment::value;

    static constexpr bool propagate_on_move =
        traits::propagate_on_container_move_assignment::value;

    static constexpr bool propagate_on_swap =
        traits::propagate_on_container_swap::value;

    static constexpr bool is_always_equal =
        traits::is_always_equal::value;

    static alloc_type select_on_copy(const alloc_type& a) {
        return traits::select_on_container_copy_construction(a);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_ALLOCATOR_SUPPORT_HPP