#include "c_api.h"
#include "inference/test_malloc_interposer.h"

#include <gtest/gtest.h>

#include <array>

namespace {

namespace test = aethermind::test;

TEST(AMError, CreateReadDestroy) {
    am_error_handle error = am_error_create(AM_STATUS_INVALID_ARGUMENT, "prompt must not be empty");
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(am_error_code(error), AM_STATUS_INVALID_ARGUMENT);
    EXPECT_STREQ(am_error_message(error), "prompt must not be empty");
    am_error_destroy(error);
}

TEST(AMError, NullMessageBecomesEmptyString) {
    am_error_handle error = am_error_create(AM_STATUS_INTERNAL, nullptr);
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(am_error_code(error), AM_STATUS_INTERNAL);
    EXPECT_STREQ(am_error_message(error), "");
    am_error_destroy(error);
}

TEST(AMError, NullHandleFallbacks) {
    EXPECT_EQ(am_error_code(nullptr), AM_STATUS_UNKNOWN);
    EXPECT_STREQ(am_error_message(nullptr), "");
    am_error_destroy(nullptr);
}

TEST(AMError, ObjectAllocationFailureReturnsNull) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }

    test::BeginMallocCallCounting();
    test::ScopedMallocFailure failure(0);
    const am_error_handle error = am_error_create(AM_STATUS_INTERNAL, "object allocation");
    const auto injected = failure.Stop();
    const auto counts = test::EndMallocCallCounting();

    EXPECT_TRUE(injected.allocation_failed);
    EXPECT_EQ(error, nullptr);
    // The failed malloc contributes one call without creating a live block.
    EXPECT_EQ(counts.malloc_calls, counts.free_calls + 1);
    am_error_destroy(error);
}

TEST(AMError, MessageAllocationFailureReleasesObject) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    std::array<char, 4096> message;
    message.fill('m');
    message.back() = '\0';

    test::BeginMallocCallCounting();
    // Match only the string storage, after the error object was allocated.
    test::ScopedMallocFailure failure(message.size());
    const am_error_handle error = am_error_create(AM_STATUS_RESOURCE_EXHAUSTED, message.data());
    const auto injected = failure.Stop();
    const auto counts = test::EndMallocCallCounting();

    EXPECT_TRUE(injected.allocation_failed);
    EXPECT_EQ(error, nullptr);
    EXPECT_GE(counts.malloc_calls, 2U);
    EXPECT_EQ(counts.malloc_calls, counts.free_calls + 1);
    am_error_destroy(error);
}

} // namespace
