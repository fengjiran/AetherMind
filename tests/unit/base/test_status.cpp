#include "aethermind/base/status.h"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

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

TEST(Status, FactoriesCoverAllPublicStatusCodes) {
    // C ABI exposes 18 status codes; the C++ API must construct all of them.
    // Verifies the 7 previously missing factories behave consistently with
    // existing ones: correct code, non-OK, ToString format, C ABI mapping.
    struct Case {
        StatusCode code;
        Status status;
        std::string_view name;
        std::string_view msg;
    };
    const Case cases[] = {
            {StatusCode::kCancelled, Status::Cancelled("c"), "CANCELLED", "c"},
            {StatusCode::kUnknown, Status::Unknown("u"), "UNKNOWN", "u"},
            {StatusCode::kDeadlineExceeded, Status::DeadlineExceeded("d"), "DEADLINE_EXCEEDED", "d"},
            {StatusCode::kAborted, Status::Aborted("a"), "ABORTED", "a"},
            {StatusCode::kUnavailable, Status::Unavailable("un"), "UNAVAILABLE", "un"},
            {StatusCode::kDataLoss, Status::DataLoss("dl"), "DATA_LOSS", "dl"},
            {StatusCode::kUnauthenticated, Status::Unauthenticated("ua"), "UNAUTHENTICATED", "ua"},
    };
    for (const auto& c: cases) {
        EXPECT_FALSE(c.status.ok()) << c.name;
        EXPECT_EQ(c.status.code(), c.code) << c.name;
        EXPECT_EQ(c.status.message(), c.msg) << c.name;
        EXPECT_EQ(c.status.ToString(), std::string(c.name) + ": " + std::string(c.msg)) << c.name;
        EXPECT_EQ(ToAMStatusCode(c.code), static_cast<am_status_code>(c.code)) << c.name;
    }
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

TEST(StatusOr, MoveAssignNoexceptTracksTMoveAssign) {
    // T's move ctor is nothrow (satisfies is_nothrow_move_constructible_v),
    // but move assignment can throw. StatusOr<T>'s move-assignment must
    // propagate the exception rather than calling std::terminate().
    struct ThrowingMoveAssign {
        int value = 0;
        ThrowingMoveAssign() = default;
        ThrowingMoveAssign(int v) noexcept : value(v) {}
        // Nothrow move ctor: satisfies the class constraint.
        ThrowingMoveAssign(ThrowingMoveAssign&& other) noexcept = default;
        ThrowingMoveAssign(const ThrowingMoveAssign&) = default;
        // Throwing move assignment: StatusOr<T>::operator=(StatusOr<T>&&) must
        // also be noexcept(false), or std::terminate() would be called.
        ThrowingMoveAssign& operator=(ThrowingMoveAssign&&) { throw std::runtime_error("throwing move assign"); }
        ThrowingMoveAssign& operator=(const ThrowingMoveAssign&) = default;
    };

    // Compile-time contract: StatusOr move-assign noexcept mirrors T's.
    static_assert(!noexcept(std::declval<StatusOr<ThrowingMoveAssign>&>() =
                                    std::declval<StatusOr<ThrowingMoveAssign>&&>()),
                  "StatusOr move-assign must be noexcept(false) when T move-assign is");

    // Runtime contract: exception propagates, no std::terminate().
    StatusOr<ThrowingMoveAssign> a(std::in_place, 1);
    StatusOr<ThrowingMoveAssign> b(std::in_place, 2);
    EXPECT_THROW({ a = std::move(b); }, std::runtime_error);
    // Source state is preserved (a still holds original value) because the
    // variant's move-assign dispatches to T's move-assign, which throws before
    // any state mutation completes.
    EXPECT_TRUE(a.ok());
    EXPECT_EQ(a->value, 1);
}

TEST(StatusOr, MoveAssignNothrowWhenTMoveAssignNothrow) {
    // For a T with nothrow move assignment, StatusOr<T>::operator=(StatusOr<T>&&)
    // must also be noexcept — the contract tracks T's, not stronger, not weaker.
    static_assert(noexcept(std::declval<StatusOr<int>&>() =
                                   std::declval<StatusOr<int>&&>()),
                  "StatusOr<int> move-assign must be noexcept because int move-assign is");

    StatusOr<int> a(std::in_place, 1);
    StatusOr<int> b(std::in_place, 2);
    a = std::move(b);
    EXPECT_TRUE(a.ok());
    EXPECT_EQ(*a, 2);
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

// ============================================================================
// Status::WithMessage contract
// ============================================================================

TEST(Status, WithMessagePreservesCodeAndReplacesMessage) {
    Status orig = Status::NotFound("file not found");
    Status augmented = orig.WithMessage("config load: file not found");
    EXPECT_EQ(augmented.code(), StatusCode::kNotFound);
    EXPECT_EQ(augmented.message(), "config load: file not found");
    // Original is not mutated: WithMessage returns a new Status.
    EXPECT_EQ(orig.message(), "file not found");
}

TEST(Status, WithMessageOnOkReturnsOkRegardlessOfMessage) {
    Status ok = Status::Ok();
    Status augmented = ok.WithMessage("attaching message to success has no semantic value");
    EXPECT_TRUE(augmented.ok());
    EXPECT_EQ(augmented.code(), StatusCode::kOk);
    EXPECT_EQ(augmented.message(), "");
}

TEST(Status, WithMessageEmptyStringProducesCodeOnlyToString) {
    Status orig = Status::Internal("original detail");
    Status augmented = orig.WithMessage("");
    EXPECT_EQ(augmented.code(), StatusCode::kInternal);
    EXPECT_EQ(augmented.message(), "");
    EXPECT_EQ(augmented.ToString(), "INTERNAL");
}

// ============================================================================
// operator==(StatusCode) contract
// ============================================================================

TEST(Status, StatusCodeEqualityComparesCodeOnly) {
    EXPECT_EQ(Status::NotFound("a"), StatusCode::kNotFound);
    EXPECT_NE(Status::NotFound("a"), StatusCode::kInternal);
    EXPECT_NE(Status::Ok(), StatusCode::kNotFound);
    EXPECT_EQ(Status::Ok(), StatusCode::kOk);
    // Different messages do not affect code comparison.
    EXPECT_EQ(Status::Internal("x"), StatusCode::kInternal);
    EXPECT_NE(Status::Internal("x"), StatusCode::kNotFound);
}

TEST(StatusOr, StatusCodeEqualityMatchesHeldStatus) {
    StatusOr<int> ok_result(42);
    EXPECT_EQ(ok_result, StatusCode::kOk);
    EXPECT_NE(ok_result, StatusCode::kInternal);

    StatusOr<int> err_result(Status::Internal("err"));
    EXPECT_EQ(err_result, StatusCode::kInternal);
    EXPECT_NE(err_result, StatusCode::kOk);
}

// ============================================================================
// Full factory contract: all 17 error codes + Ok (covered by DefaultIsOk).
// ============================================================================

TEST(Status, AllFactoriesProduceCorrectCodeAndPreserveMessage) {
    struct Case {
        StatusCode code;
        Status (*factory)(std::string_view);
        std::string_view name;
    };
    const Case cases[] = {
            {StatusCode::kCancelled, &Status::Cancelled, "CANCELLED"},
            {StatusCode::kUnknown, &Status::Unknown, "UNKNOWN"},
            {StatusCode::kInvalidArgument, &Status::InvalidArgument, "INVALID_ARGUMENT"},
            {StatusCode::kDeadlineExceeded, &Status::DeadlineExceeded, "DEADLINE_EXCEEDED"},
            {StatusCode::kNotFound, &Status::NotFound, "NOT_FOUND"},
            {StatusCode::kAlreadyExists, &Status::AlreadyExists, "ALREADY_EXISTS"},
            {StatusCode::kPermissionDenied, &Status::PermissionDenied, "PERMISSION_DENIED"},
            {StatusCode::kResourceExhausted, &Status::ResourceExhausted, "RESOURCE_EXHAUSTED"},
            {StatusCode::kFailedPrecondition, &Status::FailedPrecondition, "FAILED_PRECONDITION"},
            {StatusCode::kAborted, &Status::Aborted, "ABORTED"},
            {StatusCode::kOutOfRange, &Status::OutOfRange, "OUT_OF_RANGE"},
            {StatusCode::kUnimplemented, &Status::Unimplemented, "UNIMPLEMENTED"},
            {StatusCode::kInternal, &Status::Internal, "INTERNAL"},
            {StatusCode::kUnavailable, &Status::Unavailable, "UNAVAILABLE"},
            {StatusCode::kDataLoss, &Status::DataLoss, "DATA_LOSS"},
            {StatusCode::kUnauthenticated, &Status::Unauthenticated, "UNAUTHENTICATED"},
            {StatusCode::kOverflow, &Status::Overflow, "OVERFLOW"},
    };
    for (const auto& c: cases) {
        const std::string msg = std::string(c.name) + " detail";
        Status s = c.factory(msg);
        EXPECT_FALSE(s.ok()) << c.name;
        EXPECT_EQ(s.code(), c.code) << c.name;
        EXPECT_EQ(s.message(), msg) << c.name;
        EXPECT_EQ(s.ToString(), std::string(c.name) + ": " + msg) << c.name;
        EXPECT_EQ(ToAMStatusCode(c.code), static_cast<am_status_code>(c.code)) << c.name;
        EXPECT_EQ(StatusCodeName(c.code), c.name) << c.name;
    }
}

// ============================================================================
// StatusOr in-place construction contract
// ============================================================================

TEST(StatusOr, InPlaceConstructsValueFromArgs) {
    struct Point {
        int x, y;
        Point(int x_, int y_) : x(x_), y(y_) {}
    };
    StatusOr<Point> result(std::in_place, 3, 4);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result->x, 3);
    EXPECT_EQ(result->y, 4);
}

TEST(StatusOr, InPlaceConstructsContainerFromInitializerList) {
    // Validates the Args&&... pack handles initializer_list arguments.
    StatusOr<std::vector<int>> result(std::in_place, std::initializer_list<int>{1, 2, 3});
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0], 1);
    EXPECT_EQ((*result)[2], 3);
}

// ============================================================================
// Error-state accessor contract
// ============================================================================

TEST(StatusOr, GetIfOkReturnsNullptrOnError) {
    StatusOr<int> err(Status::Internal("boom"));
    EXPECT_EQ(err.get_if_ok(), nullptr);
    // Const overload returns const T* (nullptr on error).
    const StatusOr<int> const_err(Status::Internal("boom"));
    EXPECT_EQ(const_err.get_if_ok(), nullptr);
}

TEST(StatusOr, ValueOrPreservesValueOnOkAndFallbackOnError) {
    StatusOr<int> ok(42);
    StatusOr<int> err(Status::Internal("fail"));
    EXPECT_EQ(ok.value_or(0), 42);
    EXPECT_EQ(err.value_or(99), 99);
    // Move overload: rvalue ok moves value out; rvalue err returns fallback.
    StatusOr<int> ok_movable(7);
    EXPECT_EQ(std::move(ok_movable).value_or(0), 7);
    StatusOr<int> err_movable(Status::Internal("fail"));
    EXPECT_EQ(std::move(err_movable).value_or(11), 11);
}

// ============================================================================
// Propagation macro contracts: AM_RETURN_IF_ERROR / _WITH_MSG / AM_ASSIGN_OR_RETURN
// ============================================================================

// Helper: returns Status (distinct from GetValueOrError which returns StatusOr<int>).
Status ReturnsStatus(bool fail) {
    if (fail) return Status::Internal("status-path error");
    return Status::Ok();
}

// AM_RETURN_IF_ERROR: Status -> Status
Status ReturnIfErrorStatusToStatus(bool fail) {
    AM_RETURN_IF_ERROR(ReturnsStatus(fail));
    return Status::Ok();
}

// AM_RETURN_IF_ERROR: StatusOr<T> -> StatusOr<U>
StatusOr<int> ReturnIfErrorStatusOrToStatusOr(bool fail) {
    AM_RETURN_IF_ERROR(GetValueOrError(fail));
    return 100;
}

// AM_RETURN_IF_ERROR_WITH_MSG: Status -> Status (augmented message, preserved code)
Status ReturnIfErrorWithMsgAugmentsMessage(bool fail) {
    AM_RETURN_IF_ERROR_WITH_MSG(ReturnsStatus(fail), "context: load failed");
    return Status::Ok();
}

// AM_ASSIGN_OR_RETURN: declaration lhs (auto x)
StatusOr<int> AssignOrReturnToDeclaration(bool fail) {
    AM_ASSIGN_OR_RETURN(auto value, GetValueOrError(fail));
    return value * 2;// 42 * 2 = 84 on success
}

// AM_ASSIGN_OR_RETURN: existing variable lhs
StatusOr<int> AssignOrReturnToExistingVar(bool fail) {
    int target = -1;
    AM_ASSIGN_OR_RETURN(target, GetValueOrError(fail));
    return target;
}

TEST(StatusMacros, ReturnIfErrorPropagatesStatusToStatus) {
    EXPECT_TRUE(ReturnIfErrorStatusToStatus(false).ok());
    Status err = ReturnIfErrorStatusToStatus(true);
    EXPECT_FALSE(err.ok());
    EXPECT_EQ(err.code(), StatusCode::kInternal);
    EXPECT_EQ(err.message(), "status-path error");
}

TEST(StatusMacros, ReturnIfErrorPropagatesStatusOrToStatusOr) {
    StatusOr<int> ok = ReturnIfErrorStatusOrToStatusOr(false);
    EXPECT_TRUE(ok.ok());
    EXPECT_EQ(*ok, 100);
    StatusOr<int> err = ReturnIfErrorStatusOrToStatusOr(true);
    EXPECT_FALSE(err.ok());
    EXPECT_EQ(err.status().code(), StatusCode::kInternal);
    EXPECT_EQ(err.status().message(), "test error from StatusOr");
}

TEST(StatusMacros, ReturnIfErrorWithMsgPreservesCodeAndAugmentsMessage) {
    EXPECT_TRUE(ReturnIfErrorWithMsgAugmentsMessage(false).ok());
    Status err = ReturnIfErrorWithMsgAugmentsMessage(true);
    EXPECT_FALSE(err.ok());
    // Code is preserved from the original error.
    EXPECT_EQ(err.code(), StatusCode::kInternal);
    // Message is "<context>: <original message>".
    EXPECT_EQ(err.message(), "context: load failed: status-path error");
}

TEST(StatusMacros, AssignOrReturnToDeclarationUnwrapsOnSuccess) {
    StatusOr<int> result = AssignOrReturnToDeclaration(false);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(*result, 84);
}

TEST(StatusMacros, AssignOrReturnToExistingVarUnwrapsOnSuccess) {
    StatusOr<int> result = AssignOrReturnToExistingVar(false);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(*result, 42);
}

TEST(StatusMacros, AssignOrReturnPropagatesErrorOnFailure) {
    StatusOr<int> result = AssignOrReturnToDeclaration(true);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInternal);
    EXPECT_EQ(result.status().message(), "test error from StatusOr");
}

// ============================================================================
// Class-level [[nodiscard]] note
// ============================================================================
// class-level [[nodiscard]] on Status and StatusOr<T> cannot be verified by
// static_assert (no standard trait exposes the attribute). Enforcement is a
// compiler warning: any function returning Status/StatusOr<T> whose return
// value is discarded triggers -Wunused-result. Build with -Werror -Wunused-result
// to hard-fail on violations. The borrowing-accessor static_asserts below
// cover the complementary contract: rvalue borrowing is rejected at compile
// time to prevent dangling references/pointers into temporaries.

// Compile-time contract: borrowing accessors are blocked on rvalues to prevent
// dangling references/pointers into temporaries.
#define AM_DEFINE_ACCESSOR_TRAIT(TraitName, Expr)    \
    template<typename T, typename = void>            \
    struct TraitName : std::false_type {};           \
    template<typename T>                             \
    struct TraitName<T, std::void_t<decltype(Expr)>> \
        : std::true_type {}
AM_DEFINE_ACCESSOR_TRAIT(RvalueMessage, std::declval<T&&>().message());
AM_DEFINE_ACCESSOR_TRAIT(ConstRvalueMessage, std::declval<const T&&>().message());
AM_DEFINE_ACCESSOR_TRAIT(RvalueGetIfOk, std::declval<T&&>().get_if_ok());
AM_DEFINE_ACCESSOR_TRAIT(ConstRvalueGetIfOk, std::declval<const T&&>().get_if_ok());
AM_DEFINE_ACCESSOR_TRAIT(RvalueArrow, std::declval<T&&>().operator->());
AM_DEFINE_ACCESSOR_TRAIT(ConstRvalueArrow, std::declval<const T&&>().operator->());
AM_DEFINE_ACCESSOR_TRAIT(RvalueStatus, std::declval<T&&>().status());
AM_DEFINE_ACCESSOR_TRAIT(ConstRvalueStatus, std::declval<const T&&>().status());
AM_DEFINE_ACCESSOR_TRAIT(RvalueValue, std::declval<T&&>().value());
AM_DEFINE_ACCESSOR_TRAIT(ConstRvalueValue, std::declval<const T&&>().value());
AM_DEFINE_ACCESSOR_TRAIT(RvalueDeref, std::declval<T&&>().operator*());
AM_DEFINE_ACCESSOR_TRAIT(ConstRvalueDeref, std::declval<const T&&>().operator*());
AM_DEFINE_ACCESSOR_TRAIT(LvalueMessage, std::declval<T&>().message());
AM_DEFINE_ACCESSOR_TRAIT(ConstLvalueMessage, std::declval<const T&>().message());
AM_DEFINE_ACCESSOR_TRAIT(LvalueGetIfOk, std::declval<T&>().get_if_ok());
AM_DEFINE_ACCESSOR_TRAIT(ConstLvalueGetIfOk, std::declval<const T&>().get_if_ok());
AM_DEFINE_ACCESSOR_TRAIT(LvalueStatus, std::declval<T&>().status());
AM_DEFINE_ACCESSOR_TRAIT(ConstLvalueStatus, std::declval<const T&>().status());
AM_DEFINE_ACCESSOR_TRAIT(LvalueValue, std::declval<T&>().value());
AM_DEFINE_ACCESSOR_TRAIT(ConstLvalueValue, std::declval<const T&>().value());
#undef AM_DEFINE_ACCESSOR_TRAIT

static_assert(!RvalueMessage<Status>::value, "rvalue Status::message() must be rejected");
static_assert(!ConstRvalueMessage<Status>::value, "const rvalue Status::message() must be rejected");
static_assert(!RvalueGetIfOk<StatusOr<int>>::value, "rvalue StatusOr::get_if_ok() must be rejected");
static_assert(!ConstRvalueGetIfOk<StatusOr<int>>::value, "const rvalue StatusOr::get_if_ok() must be rejected");
static_assert(!RvalueArrow<StatusOr<int>>::value, "rvalue StatusOr::operator->() must be rejected");
static_assert(!ConstRvalueArrow<StatusOr<int>>::value, "const rvalue StatusOr::operator->() must be rejected");
// Non-const rvalue (&&) overloads are intentionally kept: status() returns
// Status by value, value()/operator*() return T&& for move semantics. They do
// not borrow from the temporary — they transfer ownership. Only const&&
// borrowing is deleted (a const rvalue cannot be moved from, so any reference
// returned from it would dangle).
static_assert(RvalueStatus<StatusOr<int>>::value, "non-const rvalue StatusOr::status() is allowed (returns Status by value)");
static_assert(!ConstRvalueStatus<StatusOr<int>>::value, "const rvalue StatusOr::status() must be rejected");
static_assert(RvalueValue<StatusOr<int>>::value, "non-const rvalue StatusOr::value() is allowed (returns T&& for move)");
static_assert(!ConstRvalueValue<StatusOr<int>>::value, "const rvalue StatusOr::value() must be rejected");
static_assert(RvalueDeref<StatusOr<int>>::value, "non-const rvalue StatusOr::operator*() is allowed (returns T&& for move)");
static_assert(!ConstRvalueDeref<StatusOr<int>>::value, "const rvalue StatusOr::operator*() must be rejected");
static_assert(LvalueMessage<Status>::value, "lvalue Status::message() must be accepted");
static_assert(ConstLvalueMessage<Status>::value, "const lvalue Status::message() must be accepted");
static_assert(LvalueGetIfOk<StatusOr<int>>::value, "lvalue StatusOr::get_if_ok() must be accepted");
static_assert(ConstLvalueGetIfOk<StatusOr<int>>::value, "const lvalue StatusOr::get_if_ok() must be accepted");
static_assert(LvalueStatus<StatusOr<int>>::value, "lvalue StatusOr::status() must be accepted");
static_assert(ConstLvalueStatus<StatusOr<int>>::value, "const lvalue StatusOr::status() must be accepted");
static_assert(LvalueValue<StatusOr<int>>::value, "lvalue StatusOr::value() must be accepted");
static_assert(ConstLvalueValue<StatusOr<int>>::value, "const lvalue StatusOr::value() must be accepted");

}// namespace
