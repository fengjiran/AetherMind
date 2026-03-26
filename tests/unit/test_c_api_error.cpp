#include "c_api.h"

#include <gtest/gtest.h>

namespace {

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

}// namespace
