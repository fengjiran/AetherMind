#include "aethermind/base/macros.h"
#include "test_malloc_interposer.h"

#include <cstdlib>
#include <gtest/gtest.h>

namespace {

using namespace aethermind::test;

TEST(MallocInterposerContract, AvailabilityMatchesBuildConfiguration) {
#if defined(AM_SANITIZE_ADDRESS) || defined(AM_SANITIZE_THREAD) || \
        defined(__SANITIZE_THREAD__) || AM_HAS_FEATURE(thread_sanitizer)
    EXPECT_FALSE(MallocInterposerAvailable());
#elif defined(__GLIBC__) && defined(__linux__)
    EXPECT_TRUE(MallocInterposerAvailable());
#else
    EXPECT_FALSE(MallocInterposerAvailable());
#endif
}

#if defined(AM_SANITIZE_ADDRESS) || defined(AM_SANITIZE_THREAD) || \
        defined(__SANITIZE_THREAD__) || AM_HAS_FEATURE(thread_sanitizer)
TEST(MallocInterposerContract, SanitizerBuildKeepsNativeAllocationAvailable) {
    ScopedMallocFailure failure(0);
    void* const allocation = std::malloc(32);
    const MallocFailureResult observed = failure.Stop();

    EXPECT_NE(allocation, nullptr);
    EXPECT_FALSE(observed.allocation_failed);
    EXPECT_EQ(observed.aligned_allocation_calls, 0U);
    std::free(allocation);
}
#endif

} // namespace
