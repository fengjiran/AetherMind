// growth_policy.hpp - GrowthPolicy for amstring capacity management
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_GROWTH_POLICY_HPP
#define AETHERMIND_AMSTRING_GROWTH_POLICY_HPP

#include "config.hpp"

#include <algorithm>
#include <cstddef>

namespace aethermind {

// Default growth policy for capacity management
// Determines how capacity grows when string exceeds current limit
struct DefaultGrowthPolicy {
    // Minimum heap capacity when transitioning from small to heap
    // Ensures reasonable initial allocation to avoid immediate re-allocation
    static constexpr std::size_t MinHeapCapacity(std::size_t required) noexcept {
        return std::max(required, config::kMinHeapCapacity);
    }

    // Calculate next capacity given old capacity and required size
    // Growth strategy: 1.5x (old + old/2) with floor
    static constexpr std::size_t NextCapacity(std::size_t old_cap,
                                              std::size_t required) noexcept {
        if (required <= old_cap) {
            return old_cap;
        }

        const std::size_t growth = old_cap / config::kGrowthFactorDenominator;
        const std::size_t candidate = old_cap + growth;

        return std::max({required, candidate, config::kMinHeapCapacity});
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_GROWTH_POLICY_HPP
