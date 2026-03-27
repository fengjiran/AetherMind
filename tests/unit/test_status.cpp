#include "aethermind/base/status.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Status, DefaultIsOk) {
    Status status;
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kOk);
    EXPECT_EQ(status.message(), "");
    EXPECT_EQ(status.ToString(), "OK");
}

TEST(Status, ErrorCarriesCodeAndMessage) {
    Status status = Status::InvalidArgument("token ids must not be empty");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "token ids must not be empty");
    EXPECT_EQ(status.ToString(), "INVALID_ARGUMENT: token ids must not be empty");
}

TEST(Status, CAbiCodeMappingRoundTrip) {
    EXPECT_EQ(aethermind::ToAMStatusCode(StatusCode::kOk), AM_STATUS_OK);
    EXPECT_EQ(aethermind::ToAMStatusCode(StatusCode::kInvalidArgument), AM_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(aethermind::ToAMStatusCode(StatusCode::kInternal), AM_STATUS_INTERNAL);

    EXPECT_EQ(aethermind::FromAMStatusCode(AM_STATUS_OK), StatusCode::kOk);
    EXPECT_EQ(aethermind::FromAMStatusCode(AM_STATUS_NOT_FOUND), StatusCode::kNotFound);
    EXPECT_EQ(aethermind::FromAMStatusCode(AM_STATUS_UNAUTHENTICATED), StatusCode::kUnauthenticated);
}

TEST(StatusOr, HoldsValue) {
    StatusOr<int> result(42);
    EXPECT_TRUE(result.ok());
    ASSERT_NE(result.get_if_ok(), nullptr);
    EXPECT_EQ(*result.get_if_ok(), 42);
    EXPECT_EQ(result.status().code(), StatusCode::kOk);
    EXPECT_EQ(result.value(), 42);
    EXPECT_EQ(*result, 42);
}

TEST(StatusOr, HoldsErrorStatus) {
    StatusOr<int> result(Status::NotFound("weight tensor missing"));
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.get_if_ok(), nullptr);
    EXPECT_EQ(result.status().code(), StatusCode::kNotFound);
    EXPECT_EQ(result.status().message(), "weight tensor missing");
}

TEST(StatusOr, ValueOrFallback) {
    StatusOr<int> ok_result(11);
    StatusOr<int> err_result(Status::Internal("decode failed"));
    EXPECT_EQ(ok_result.value_or(99), 11);
    EXPECT_EQ(err_result.value_or(99), 99);
}

TEST(StatusOr, ValueThrowsWhenNotOk) {
    StatusOr<int> result(Status::Internal("decode failed"));
    EXPECT_THROW(UNUSED(result.value()), std::logic_error);
}

TEST(StatusOr, MoveOnlyType) {
    StatusOr<std::unique_ptr<int>> result(std::make_unique<int>(7));
    EXPECT_TRUE(result.ok());
    EXPECT_NE(result->get(), nullptr);
    EXPECT_EQ(*result->get(), 7);
}

TEST(StatusOr, MoveOutValue) {
    StatusOr<std::unique_ptr<int>> result(std::make_unique<int>(9));
    std::unique_ptr<int> moved = std::move(result).value();
    ASSERT_NE(moved, nullptr);
    EXPECT_EQ(*moved, 9);
}

TEST(StatusOr, RejectsOkStatusInErrorCtor) {
    EXPECT_THROW(UNUSED(StatusOr<int>(Status::Ok())), std::invalid_argument);
}

}// namespace
