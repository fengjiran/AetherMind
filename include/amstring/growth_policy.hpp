#ifndef AMSTRING_GROWTH_POLICY_HPP
#define AMSTRING_GROWTH_POLICY_HPP

#include <cstddef>

namespace aethermind {

// Growth policy for heap capacity expansion
// First version: simple geometric growth (1.5x or 2x)

class DefaultGrowthPolicy {
public:
    // Calculate new capacity when current capacity is insufficient
    // Strategy: grow by factor of 2 for small sizes, 1.5x for larger
    static constexpr size_t grow(size_t current, size_t required) {
        if (required <= current) {
            return current;
        }

        // Minimum growth: at least required
        if (current == 0) {
            return required;
        }

        // Geometric growth
        size_t geometric = current * 2;
        return geometric > required ? geometric : required;
    }

    // Maximum allowed capacity
    static constexpr size_t max_capacity() {
        return size_t(-1) / 2;// Conservative limit
    }
};

}// namespace aethermind

#endif// AMSTRING_GROWTH_POLICY_HPP