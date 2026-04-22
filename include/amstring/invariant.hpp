#ifndef AMSTRING_INVARIANT_HPP
#define AMSTRING_INVARIANT_HPP

#include "config.hpp"

namespace aethermind {

// Invariant checking for basic_string_core
// Core invariants that must hold at all times:
//
// 1. data() != nullptr
// 2. data()[size()] == CharT{}  (null terminator)
// 3. size() <= capacity()
// 4. begin() == data()
// 5. end() == data() + size()
// 6. moved-from object is valid empty string
// 7. small/heap state is O(1) distinguishable
// 8. heap: data() points to valid allocated region
// 9. heap: capacity() returns character count (not byte count)
// 10. capacity does not contain category bits

#if AMSTRING_CHECK_INVARIANTS

template<typename Core>
void check_invariants(const Core& core) {
    // TODO: Implement actual invariant checks in Milestone 1
    // This is a placeholder for now
}

#define AMSTRING_INVARIANT_CHECK(core) check_invariants(core)

#else

#define AMSTRING_INVARIANT_CHECK(core) ((void) 0)

#endif

}// namespace aethermind

#endif// AMSTRING_INVARIANT_HPP