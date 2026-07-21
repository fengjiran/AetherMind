#include "aethermind/base/status.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace {
using namespace aethermind;

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

TEST(Status, EqualityComparesCodeAndMessage) {
    EXPECT_EQ(Status::Ok(), Status::Ok());
    EXPECT_EQ(Status::InvalidArgument("bad input"), Status::InvalidArgument("bad input"));
    EXPECT_NE(Status::InvalidArgument("bad input"), Status::InvalidArgument("different input"));
    EXPECT_NE(Status::InvalidArgument("bad input"), Status::Internal("bad input"));
}

TEST(Status, CAbiCodeMappingRoundTrip) {
    EXPECT_EQ(aethermind::ToAMStatusCode(StatusCode::kOk), AM_STATUS_OK);
    EXPECT_EQ(aethermind::ToAMStatusCode(StatusCode::kInvalidArgument), AM_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(aethermind::ToAMStatusCode(StatusCode::kInternal), AM_STATUS_INTERNAL);

    EXPECT_EQ(aethermind::FromAMStatusCode(AM_STATUS_OK), StatusCode::kOk);
    EXPECT_EQ(aethermind::FromAMStatusCode(AM_STATUS_NOT_FOUND), StatusCode::kNotFound);
    EXPECT_EQ(aethermind::FromAMStatusCode(AM_STATUS_UNAUTHENTICATED), StatusCode::kUnauthenticated);
}

TEST(Status, CAbiCodeMappingRejectsOutOfRangeInput) {
    EXPECT_EQ(aethermind::FromAMStatusCode(static_cast<am_status_code>(-1)), StatusCode::kUnknown);
    EXPECT_EQ(aethermind::FromAMStatusCode(static_cast<am_status_code>(99)), StatusCode::kUnknown);
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

TEST(StatusOr, MoveOutValueViaRvalueDereference) {
    static_assert(std::is_same_v<decltype(*std::declval<StatusOr<std::unique_ptr<int>>&&>()),
                                 std::unique_ptr<int>&&>);

    StatusOr<std::unique_ptr<int>> result(std::make_unique<int>(13));
    std::unique_ptr<int> moved = std::move(*std::move(result));
    ASSERT_NE(moved, nullptr);
    EXPECT_EQ(*moved, 13);
}

// Rvalue operator->() is deleted by design: it would export a pointer into
// a temporary StatusOr, producing a dangling pointer. Use operator* (which
// returns a T&& rvalue, safe for immediate move) instead.
// See static_assert block at end of file for the compile-time contract.

TEST(StatusOr, RejectsOkStatusInErrorCtor) {
    EXPECT_THROW(UNUSED(StatusOr<int>(Status::Ok())), std::invalid_argument);
}

StatusOr<int> GetValueOrError(bool fail) {
    if (fail) {
        return Status::Internal("test error from StatusOr");
    }
    return 42;
}

Status ProcessStatusOrResult(bool fail) {
    AM_RETURN_IF_ERROR(GetValueOrError(fail));
    return Status::Ok();
}

TEST(StatusMacros, PropagatesErrorFromStatusOrToStatus) {
    Status success_status = ProcessStatusOrResult(/*fail=*/false);
    EXPECT_TRUE(success_status.ok());
    EXPECT_EQ(success_status.code(), StatusCode::kOk);

    Status error_status = ProcessStatusOrResult(/*fail=*/true);
    EXPECT_FALSE(error_status.ok());
    EXPECT_EQ(error_status.code(), StatusCode::kInternal);
    EXPECT_EQ(error_status.message(), "test error from StatusOr");
}

// Compile-time contract: borrowing accessors are blocked on rvalues to prevent
// dangling references/pointers into temporaries.
#define AM_DEFINE_ACCESSOR_TRAIT(TraitName, Expr)    \
    template<typename T, typename = void>            \
    struct TraitName : std::false_type {};           \
    template<typename T>                             \
    struct TraitName<T, std::void_t<decltype(Expr)>> \
        : std::true_type {}
AM_DEFINE_ACCESSOR_TRAIT(RvalueMessage, std::declval<T&&>().message());
AM_DEFINE_ACCESSOR_TRAIT(RvalueGetIfOk, std::declval<T&&>().get_if_ok());
AM_DEFINE_ACCESSOR_TRAIT(RvalueArrow, std::declval<T&&>().operator->());
AM_DEFINE_ACCESSOR_TRAIT(LvalueMessage, std::declval<T&>().message());
AM_DEFINE_ACCESSOR_TRAIT(ConstLvalueMessage, std::declval<const T&>().message());
AM_DEFINE_ACCESSOR_TRAIT(LvalueGetIfOk, std::declval<T&>().get_if_ok());
AM_DEFINE_ACCESSOR_TRAIT(ConstLvalueGetIfOk, std::declval<const T&>().get_if_ok());
#undef AM_DEFINE_ACCESSOR_TRAIT

static_assert(!RvalueMessage<Status>::value, "rvalue Status::message() must be rejected");
static_assert(!RvalueGetIfOk<StatusOr<int>>::value, "rvalue StatusOr::get_if_ok() must be rejected");
static_assert(!RvalueArrow<StatusOr<int>>::value, "rvalue StatusOr::operator->() must be rejected");
static_assert(LvalueMessage<Status>::value, "lvalue Status::message() must be accepted");
static_assert(ConstLvalueMessage<Status>::value, "const lvalue Status::message() must be accepted");
static_assert(LvalueGetIfOk<StatusOr<int>>::value, "lvalue StatusOr::get_if_ok() must be accepted");
static_assert(ConstLvalueGetIfOk<StatusOr<int>>::value, "const lvalue StatusOr::get_if_ok() must be accepted");

}// namespace
