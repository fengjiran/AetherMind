#ifndef AETHERMIND_TEST_UTILS_TEST_ALLOCATOR_H
#define AETHERMIND_TEST_UTILS_TEST_ALLOCATOR_H

#include "aethermind/base/device.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/memory/cpu_allocator.h"

namespace aethermind::test {

/// @brief Process-wide CPU allocator for tests that only need managed storage.
///
/// Shared so a test does not have to construct one, and deliberately
/// long-lived: every Buffer taken from it is released before the allocator dies.
/// A test that needs to observe release-after-destruction, inspect ledger
/// counters, or control CPUAllocatorPolicy must construct its own CPUAllocator
/// instead of using this one.
inline Allocator& TestCpuAllocator() {
    static CPUAllocator allocator(Device::CPU());
    return allocator;
}

} // namespace aethermind::test

#endif // AETHERMIND_TEST_UTILS_TEST_ALLOCATOR_H
