#ifndef AMSTRING_CORE_HPP
#define AMSTRING_CORE_HPP

#include "category.hpp"
#include "config.hpp"
#include "growth_policy.hpp"
#include "invariant.hpp"
#include "layout.hpp"

#include <memory>

namespace aethermind {

// Core storage implementation for Small String Optimization (SSO)
// First version: generic implementation for all CharT
// Later: char-optimized specialization (Milestone 13)
template<typename CharT,
         typename Traits = std::char_traits<CharT>,
         typename Allocator = std::allocator<CharT>>
class basic_string_core {
public:
    using value_type = CharT;
    using traits_type = Traits;
    using allocator_type = Allocator;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

    // TODO: Implement in Milestone 1
    // This is a skeleton for M0

private:
    Storage<CharT> storage_;
    Allocator allocator_;// EBO optimization for std::allocator

    // Constants
    static constexpr size_type kSmallCapacity = Storage<CharT>::kSmallCapacity;
    static constexpr size_type kMaxSmallSize = kSmallCapacity;
};

}// namespace aethermind

#endif// AMSTRING_CORE_HPP