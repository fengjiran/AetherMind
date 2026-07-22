// Copyright 2026 The AetherMind Authors
// SPDX-License-Identifier: Apache-2.0
//
// Status and StatusOr for error handling in Phase 1 runtime.
//
// This module provides a contract/result error model similar to absl::Status
// and gRPC status codes. It is the primary error handling mechanism for the
// inference runtime, replacing exception-heavy paths with explicit status checks.
//
// Key components:
// - StatusCode: Enum of error codes aligned with gRPC and am_status_code (C ABI)
// - Status: Lightweight error container (code + message)
// - StatusOr<T>: Union of value or error, for functions that may fail
//
// Thread safety: const methods are thread-safe. StatusOr is not thread-safe
// for concurrent mutation.
//
// Dependencies: c_api.h for am_status_code, macros.h for AM_NODISCARD

#ifndef AETHERMIND_AETHERMIND_BASE_STATUS_H
#define AETHERMIND_AETHERMIND_BASE_STATUS_H

#include "c_api.h"
#include "macros.h"

#include <concepts>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace aethermind {

/// Status codes for operation results, aligned with gRPC status codes.
///
/// Invariant: numeric values map 1:1 to am_status_code (C ABI). The mapping
/// is verified by static_asserts below the enum definition, so adding a new
/// code requires extending both this enum and c_api.h in lockstep.
enum class StatusCode : uint8_t {
    kOk = 0,
    kCancelled,
    kUnknown,
    kInvalidArgument,
    kDeadlineExceeded,
    kNotFound,
    kAlreadyExists,
    kPermissionDenied,
    kResourceExhausted,
    kFailedPrecondition,
    kAborted,
    kOutOfRange,
    kUnimplemented,
    kInternal,
    kUnavailable,
    kDataLoss,
    kOverflow,
    kUnauthenticated
};

static_assert(static_cast<uint8_t>(StatusCode::kOk) == AM_STATUS_OK, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kCancelled) == AM_STATUS_CANCELLED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kUnknown) == AM_STATUS_UNKNOWN, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kInvalidArgument) == AM_STATUS_INVALID_ARGUMENT, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kDeadlineExceeded) == AM_STATUS_DEADLINE_EXCEEDED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kNotFound) == AM_STATUS_NOT_FOUND, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kAlreadyExists) == AM_STATUS_ALREADY_EXISTS, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kPermissionDenied) == AM_STATUS_PERMISSION_DENIED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kResourceExhausted) == AM_STATUS_RESOURCE_EXHAUSTED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kFailedPrecondition) == AM_STATUS_FAILED_PRECONDITION, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kAborted) == AM_STATUS_ABORTED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kOutOfRange) == AM_STATUS_OUT_OF_RANGE, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kUnimplemented) == AM_STATUS_UNIMPLEMENTED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kInternal) == AM_STATUS_INTERNAL, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kUnavailable) == AM_STATUS_UNAVAILABLE, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kDataLoss) == AM_STATUS_DATA_LOSS, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kUnauthenticated) == AM_STATUS_UNAUTHENTICATED, "Status code mismatch!");
static_assert(static_cast<uint8_t>(StatusCode::kOverflow) == AM_STATUS_OVERFLOW, "Status code mismatch!");

/// Returns the human-readable name for a status code.
///
/// For example, StatusCode::kNotFound returns "NOT_FOUND".
/// Guaranteed to return a valid string_view for all defined StatusCode values.
AM_NODISCARD constexpr std::string_view StatusCodeName(StatusCode code) noexcept {
    switch (code) {
        case StatusCode::kOk:
            return "OK";
        case StatusCode::kCancelled:
            return "CANCELLED";
        case StatusCode::kUnknown:
            return "UNKNOWN";
        case StatusCode::kInvalidArgument:
            return "INVALID_ARGUMENT";
        case StatusCode::kDeadlineExceeded:
            return "DEADLINE_EXCEEDED";
        case StatusCode::kNotFound:
            return "NOT_FOUND";
        case StatusCode::kAlreadyExists:
            return "ALREADY_EXISTS";
        case StatusCode::kPermissionDenied:
            return "PERMISSION_DENIED";
        case StatusCode::kResourceExhausted:
            return "RESOURCE_EXHAUSTED";
        case StatusCode::kFailedPrecondition:
            return "FAILED_PRECONDITION";
        case StatusCode::kAborted:
            return "ABORTED";
        case StatusCode::kOutOfRange:
            return "OUT_OF_RANGE";
        case StatusCode::kUnimplemented:
            return "UNIMPLEMENTED";
        case StatusCode::kInternal:
            return "INTERNAL";
        case StatusCode::kUnavailable:
            return "UNAVAILABLE";
        case StatusCode::kDataLoss:
            return "DATA_LOSS";
        case StatusCode::kUnauthenticated:
            return "UNAUTHENTICATED";
        case StatusCode::kOverflow:
            return "OVERFLOW";
    }
    // Defensive return for compilers that cannot prove the switch is exhaustive.
    // Unreachable for any value within the StatusCode enum.
    return "UNKNOWN";
}

/// Converts a StatusCode to its C ABI equivalent.
///
/// The conversion is a direct numerical cast. Every StatusCode value maps to a
/// valid am_status_code, verified by static_assert at the enum definition.
AM_NODISCARD constexpr am_status_code ToAMStatusCode(StatusCode code) noexcept {
    return static_cast<am_status_code>(code);
}

/// Converts a C ABI status code back to StatusCode.
///
/// Out-of-range values (outside [AM_STATUS_OK, AM_STATUS_UNAUTHENTICATED]) are
/// mapped to StatusCode::kUnknown. This prevents undefined behavior from
/// casting an invalid integer to the StatusCode enum.
AM_NODISCARD constexpr StatusCode FromAMStatusCode(am_status_code code) noexcept {
    if (code < AM_STATUS_OK || code > AM_STATUS_UNAUTHENTICATED) {
        return StatusCode::kUnknown;
    }
    return static_cast<StatusCode>(code);
}

/// Represents the result of an operation that may fail.
///
/// Modeled after absl::Status and gRPC status. Default-constructed Status is OK.
/// Thread-compatible (const methods are thread-safe); non-const mutation is not.
///
/// Invariant: when code_ == kOk, message_ is empty. Construction is restricted
/// to the public factories and the private code+message ctor below; the
/// factories enforce the invariant by only attaching messages to non-OK codes.
///
/// @note Use StatusOr<T> for operations that return a value on success.
class AM_NODISCARD Status {
public:
    /// Creates an OK status.
    Status() noexcept : code_(StatusCode::kOk) {}

    /// Returns an OK status.
    ///
    /// Named factory equivalent to default construction. Idiomatic in return
    /// statements: `return Status::Ok();`.
    static Status Ok() noexcept {
        return {};
    }

    static Status Cancelled(std::string_view message) {
        return Status(StatusCode::kCancelled, std::string(message));
    }

    static Status Unknown(std::string_view message) {
        return Status(StatusCode::kUnknown, std::string(message));
    }

    static Status DeadlineExceeded(std::string_view message) {
        return Status(StatusCode::kDeadlineExceeded, std::string(message));
    }

    static Status Aborted(std::string_view message) {
        return Status(StatusCode::kAborted, std::string(message));
    }

    static Status Unavailable(std::string_view message) {
        return Status(StatusCode::kUnavailable, std::string(message));
    }

    static Status DataLoss(std::string_view message) {
        return Status(StatusCode::kDataLoss, std::string(message));
    }

    static Status Unauthenticated(std::string_view message) {
        return Status(StatusCode::kUnauthenticated, std::string(message));
    }

    static Status InvalidArgument(std::string_view message) {
        return Status(StatusCode::kInvalidArgument, std::string(message));
    }

    static Status NotFound(std::string_view message) {
        return Status(StatusCode::kNotFound, std::string(message));
    }

    static Status Internal(std::string_view message) {
        return Status(StatusCode::kInternal, std::string(message));
    }

    static Status PermissionDenied(std::string_view message) {
        return Status(StatusCode::kPermissionDenied, std::string(message));
    }

    static Status AlreadyExists(std::string_view message) {
        return Status(StatusCode::kAlreadyExists, std::string(message));
    }

    static Status OutOfRange(std::string_view message) {
        return Status(StatusCode::kOutOfRange, std::string(message));
    }

    static Status FailedPrecondition(std::string_view message) {
        return Status(StatusCode::kFailedPrecondition, std::string(message));
    }

    static Status Unimplemented(std::string_view message) {
        return Status(StatusCode::kUnimplemented, std::string(message));
    }

    static Status ResourceExhausted(std::string_view message) {
        return Status(StatusCode::kResourceExhausted, std::string(message));
    }

    static Status Overflow(std::string_view message) {
        return Status(StatusCode::kOverflow, std::string(message));
    }

    /// Returns true if the status code is kOk.
    AM_NODISCARD bool ok() const noexcept {
        return code_ == StatusCode::kOk;
    }

    /// Returns true if the status is OK.
    ///
    /// Same as ok(). Enables idiomatic checks: `if (status) { ... }`.
    AM_NODISCARD explicit operator bool() const noexcept {
        return ok();
    }

    /// Returns the status code.
    AM_NODISCARD StatusCode code() const noexcept {
        return code_;
    }

    /// Returns the error message.
    ///
    /// Empty string when the status is OK. Content is meaningful only when
    /// `!ok()`.
    ///
    /// Ref-qualified on lvalue `*this` to prevent binding the returned reference
    /// to a temporary: `const std::string& s = MakeStatus().message();` would
    /// dangle. Rvalue Status must be materialized first.
    AM_NODISCARD const std::string& message() & noexcept {
        return message_;
    }

    AM_NODISCARD const std::string& message() const& noexcept {
        return message_;
    }

    // Deleted to prevent exporting a reference into a temporary. Calling
    // message() on a rvalue Status would bind the returned const std::string&
    // to the temporary's internal member, producing a dangling reference once
    // the temporary is destroyed at the end of the full expression.
    //
    // Both const&& and non-const && are rejected:
    //  - A const rvalue cannot be moved from, so any reference returned from
    //    it would dangle (const&& would bind to const& if not deleted).
    //  - A non-const rvalue should be consumed via move semantics rather than
    //    borrowed; callers materializing the Status into an lvalue first can
    //    still use the lvalue overloads above.
    AM_NODISCARD const std::string& message() && noexcept = delete;
    AM_NODISCARD const std::string& message() const&& noexcept = delete;

    /// Returns a new Status with a different message, preserving the code.
    ///
    /// Does not mutate this Status. Returns Ok() when the current status is OK,
    /// regardless of the provided message — attaching a message to success has
    /// no semantic value.
    Status WithMessage(std::string message) const noexcept {
        if (ok()) {
            return Ok();
        }
        return Status(code_, std::move(message));
    }

    /// Returns a human-readable string like "NOT_FOUND: file not found".
    ///
    /// Format: "STATUS_CODE_NAME" for OK or empty-message errors, or
    /// "STATUS_CODE_NAME: message" when a message is present.
    AM_NODISCARD std::string ToString() const {
        if (ok()) {
            return std::string(StatusCodeName(StatusCode::kOk));
        }

        if (message_.empty()) {
            return std::string(StatusCodeName(code_));
        }

        return std::string(StatusCodeName(code_)) + ": " + message_;
    }

    /// Returns true if both code and message are equal.
    AM_NODISCARD bool operator==(const Status& other) const {
        return code_ == other.code_ && message_ == other.message_;
    }

    /// Returns true if code or message differ.
    AM_NODISCARD bool operator!=(const Status& other) const {
        return !(*this == other);
    }

    /// Compares the status code directly.
    ///
    /// Enables idiomatic checks: `if (status == StatusCode::kNotFound)`.
    AM_NODISCARD bool operator==(StatusCode code) const noexcept {
        return code_ == code;
    }

    /// Negated StatusCode comparison.
    AM_NODISCARD bool operator!=(StatusCode code) const noexcept {
        return code_ != code;
    }

private:
    // Private so callers must use the factories above. The factories are the
    // only public path to a non-OK Status and keep code/message consistent.
    explicit Status(StatusCode code, std::string message) noexcept
        : code_(code), message_(std::move(message)) {}

    // Invariant: empty iff code_ == kOk (see class comment).
    StatusCode code_;
    // Invariant: non-empty only when code_ != kOk.
    std::string message_;
};

template<typename T>
class StatusOr;

namespace detail {

inline Status ExtractStatus(Status&& status) noexcept;

template<typename T>
Status ExtractStatus(StatusOr<T>&& result) noexcept;

}// namespace detail

/// Evaluates expr and returns its status if not OK.
///
/// @param expr Expression returning Status or StatusOr<T>. The extracted
///             status is returned from the enclosing function, whose return
///             type must be convertible from Status (Status or StatusOr<U>).
///
/// The expression is evaluated exactly once.
///
/// Example (Status-returning function consuming a StatusOr<int> expression):
///   StatusOr<int> FetchValue();
///   Status Process() {
///     AM_RETURN_IF_ERROR(FetchValue());
///     return Status::Ok();
///   }
///
#define AM_RETURN_IF_ERROR(expr)                                                          \
    do {                                                                                  \
        auto am_status_result_ = (expr);                                                  \
        if (!am_status_result_.ok()) AM_UNLIKELY {                                        \
                return ::aethermind::detail::ExtractStatus(std::move(am_status_result_)); \
            }                                                                             \
    } while (false)

/// Evaluates expr and returns an augmented error status if not OK.
///
/// The returned Status preserves the original error code; only the message
/// is replaced with "<msg>: <original message>".
///
/// @warning Only use in functions returning Status. If the enclosing function
///          returns StatusOr<T>, this macro will fail to compile because
///          Status cannot implicitly convert to StatusOr<T>. Use
///          AM_RETURN_IF_ERROR and construct the StatusOr error separately.
///
/// @param expr Expression returning Status.
/// @param msg  Prefix prepended to the original error message (joined by ": ").
///
/// Example:
///   Status LoadModel(const std::string& path) {
///     AM_RETURN_IF_ERROR_WITH_MSG(OpenFile(path), "Failed to load model");
///     // If OpenFile returned Status(NOT_FOUND, "file not found"), returns:
///     //   Status(NOT_FOUND, "Failed to load model: file not found")
///     return Status::Ok();
///   }
#define AM_RETURN_IF_ERROR_WITH_MSG(expr, msg)                                \
    do {                                                                      \
        auto am_status_result_ = (expr);                                      \
        if (!am_status_result_.ok()) AM_UNLIKELY {                            \
                const std::string am_orig_msg_ = am_status_result_.message(); \
                return std::move(am_status_result_)                           \
                        .WithMessage(std::string(msg) + ": " + am_orig_msg_); \
            }                                                                 \
    } while (false)

/// Unwraps a StatusOr<T> into lhs, propagating the error on failure.
///
/// Evaluates expr exactly once. If the result is not OK, returns its status
/// from the enclosing function. Otherwise, moves the value into lhs.
///
/// lhs may be a declaration (e.g. `auto x` or `const T x`) or an assignment
/// target (an existing variable). Each invocation expands to a uniquely-named
/// temporary via __COUNTER__, so multiple AM_ASSIGN_OR_RETURN calls may appear
/// in the same scope.
///
/// Example:
///   StatusOr<int> ParseInt(const std::string& s);
///   StatusOr<size_t> Run(const std::string& s) {
///     AM_ASSIGN_OR_RETURN(auto value, ParseInt(s));
///     return static_cast<size_t>(value);
///   }
#define AM_ASSIGN_OR_RETURN(lhs, expr)                        \
    AM_ASSIGN_OR_RETURN_IMPL(                                 \
            AM_STR_CONCAT(am_assign_or_result_, __COUNTER__), \
            lhs, expr)

#define AM_ASSIGN_OR_RETURN_IMPL(result_var, lhs, expr) \
    auto result_var = (expr);                           \
    if (!result_var.ok()) AM_UNLIKELY {                 \
            return result_var.status();                 \
        }                                               \
    lhs = std::move(*result_var)

/// Represents either a value of type T or an error Status.
///
/// Used for operations that return a value on success. Never holds Status::Ok().
/// Template type T must not be a reference type or Status.
///
/// Invariant: storage_ holds either T (ok state) or a non-OK Status (error
/// state), never Status::Ok(). The error constructors enforce this by throwing
/// std::invalid_argument on an OK Status input.
///
/// Access semantics: borrowing accessors (get_if_ok, operator->, operator*,
/// value const&) are ref-qualified on lvalue *this to prevent exporting
/// references/pointers into temporaries; rvalue overloads are deleted. Move
/// accessors (value &&, operator* &&, status &&) transfer ownership and are
/// safe on rvalues.
///
/// @tparam T The value type returned on success. Must be move-constructible.
///
/// Example:
///   StatusOr<int> ParseInt(const std::string& s);
///   auto result = ParseInt("42");
///   if (result.ok()) { use(*result); } else { handle(result.status()); }
template<typename T>
class AM_NODISCARD StatusOr {
public:
    /// Static assertions ensure T is a valid type for StatusOr.
    static_assert(!std::is_reference_v<T>, "StatusOr<T&> is not supported");
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>, "StatusOr<Status> is not supported");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "StatusOr<T> requires T to be nothrow move-constructible for noexcept guarantee");

    /// Constructs a StatusOr holding a valid value from a compatible type.
    ///
    /// @tparam U Must be convertible to T and must not be Status or StatusOr,
    ///           preventing accidental hijacking of the error constructor.
    template<typename U = T>
        requires(!std::same_as<std::decay_t<U>, Status>) &&
                (!std::same_as<std::decay_t<U>, StatusOr>) &&
                std::convertible_to<U, T>
    StatusOr(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>)
        : storage_(std::in_place_type<T>, std::forward<U>(value)) {}// NOLINT(google-explicit-constructor)

    /// Constructs a StatusOr by constructing T in-place from the given arguments.
    ///
    /// @tparam Args Constructor argument types for T.
    template<typename... Args>
    StatusOr(std::in_place_t, Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)// NOLINT(google-explicit-constructor)
        : storage_(std::in_place_type<T>, std::forward<Args>(args)...) {}

    /// Constructs from an error status. Throws std::invalid_argument if status.ok().
    ///
    /// @throws std::invalid_argument if status is OK (prevents accidentally wrapping success as error).
    StatusOr(const Status& status) : storage_(status) {
        if (status.ok()) {
            throw std::invalid_argument("StatusOr error constructor requires non-OK status");
        }
    }

    StatusOr(Status&& status) : storage_(std::move(status)) {// NOLINT(google-explicit-constructor)
        if (std::get<Status>(storage_).ok()) {
            throw std::invalid_argument("StatusOr error constructor requires non-OK status");
        }
    }

    StatusOr(const StatusOr&) = default;
    // noexcept is backed by the is_nothrow_move_constructible_v<T> static_assert.
    StatusOr(StatusOr&&) noexcept = default;
    StatusOr& operator=(const StatusOr&) = default;
    // noexcept is intentionally omitted: the template constrains only T's move
    // ctor, not its move assignment. Letting the compiler derive noexcept from
    // std::variant<T, Status>::operator=(variant&&) keeps the contract honest —
    // if T's move assignment can throw, StatusOr's move assignment can throw
    // too, instead of calling std::terminate().
    StatusOr& operator=(StatusOr&&) = default;
    ~StatusOr() = default;

    /// Returns true if this StatusOr holds a value (not an error).
    AM_NODISCARD bool ok() const noexcept {
        return std::holds_alternative<T>(storage_);
    }

    /// Returns a pointer to the held value, or nullptr on error.
    ///
    /// Ref-qualified on lvalue `*this` to prevent exporting a pointer into a
    /// temporary: `const T* p = StatusOr<T>(v).get_if_ok();` would dangle.
    AM_NODISCARD const T* get_if_ok() const& noexcept {
        return ok() ? &std::get<T>(storage_) : nullptr;
    }

    AM_NODISCARD T* get_if_ok() & noexcept {
        return ok() ? &std::get<T>(storage_) : nullptr;
    }

    AM_NODISCARD const T* get_if_ok() const&& noexcept = delete;
    AM_NODISCARD T* get_if_ok() && noexcept = delete;

    AM_NODISCARD const Status& status() const& noexcept {
        // Shared static OK instance avoids per-call construction while keeping
        // the return-by-reference contract. Safe because Status::Ok() is
        // thread-compatible and immutable.
        static const Status kOkStatus = Status::Ok();
        if (ok()) {
            return kOkStatus;
        }
        return std::get<Status>(storage_);
    }

    AM_NODISCARD const Status& status() const&& noexcept = delete;

    Status status() && noexcept {
        if (ok()) {
            return Status::Ok();
        }
        return std::move(std::get<Status>(storage_));
    }

    /// Returns a const reference to the value. Throws if !ok().
    ///
    /// @throws std::logic_error if this StatusOr holds an error.
    /// @note Prefer operator* or get_if_ok() for non-throwing access.
    AM_NODISCARD const T& value() const& {
        if (!ok()) {
            throw std::logic_error("Attempted to access value() on error StatusOr: " +
                                   status().ToString());
        }
        return std::get<T>(storage_);
    }

    AM_NODISCARD const T& value() const&& = delete;

    AM_NODISCARD T& value() & {
        if (!ok()) {
            throw std::logic_error("Attempted to access value() on error StatusOr: " +
                                   status().ToString());
        }
        return std::get<T>(storage_);
    }

    AM_NODISCARD T&& value() && {
        if (!ok()) {
            throw std::logic_error("Attempted to access value() on error StatusOr: " +
                                   status().ToString());
        }
        return std::move(std::get<T>(storage_));
    }

    /// Returns the value if ok(), otherwise returns fallback.
    ///
    /// @tparam U Type convertible to T.
    /// @param fallback Value to return if this StatusOr holds an error.
    /// @return The contained value or fallback.
    template<typename U>
    AM_NODISCARD T value_or(U&& fallback) const& {
        if (ok()) {
            return std::get<T>(storage_);
        }
        return T(std::forward<U>(fallback));
    }

    template<typename U>
    AM_NODISCARD T value_or(U&& fallback) && {
        if (ok()) {
            return std::move(std::get<T>(storage_));
        }
        return T(std::forward<U>(fallback));
    }

    AM_NODISCARD const T& operator*() const& {
        return std::get<T>(storage_);
    }

    AM_NODISCARD T& operator*() & {
        return std::get<T>(storage_);
    }

    AM_NODISCARD const T&& operator*() const&& = delete;

    AM_NODISCARD T&& operator*() && {
        return std::move(std::get<T>(storage_));
    }

    AM_NODISCARD const T* operator->() const& {
        return &std::get<T>(storage_);
    }

    AM_NODISCARD T* operator->() & {
        return &std::get<T>(storage_);
    }

    AM_NODISCARD const T* operator->() const&& = delete;
    AM_NODISCARD T* operator->() && = delete;

    /// Returns true if the held status code matches.
    ///
    /// Enables idiomatic checks: `if (result == StatusCode::kNotFound)`.
    AM_NODISCARD bool operator==(StatusCode code) const noexcept {
        return status().code() == code;
    }

    /// Returns true if the held status code does not match.
    AM_NODISCARD bool operator!=(StatusCode code) const noexcept {
        return status().code() != code;
    }

private:
    // Invariant: never holds Status::Ok(). Error Status is always non-OK.
    std::variant<T, Status> storage_;
};

// Status is nothrow move-constructible and move-assignable. These are
// prerequisites for the StatusOr move-contract tracking below — if Status
// ever loses nothrow move semantics, StatusOr<T>'s move noexcept would no
// longer track T alone.
static_assert(std::is_nothrow_move_constructible_v<Status>,
              "Status must be nothrow move-constructible (StatusOr move-contract prerequisite)");
static_assert(std::is_nothrow_move_assignable_v<Status>,
              "Status must be nothrow move-assignable (StatusOr move-contract prerequisite)");

// Compile-time contract: StatusOr<T>'s move-ctor noexcept tracks T's move-ctor
// noexcept (via std::variant). Mirrors kStatusOrMoveAssignTracksT below.
template<typename T>
constexpr bool kStatusOrMoveCtorTracksT =
        std::is_nothrow_move_constructible_v<StatusOr<T>> ==
        std::is_nothrow_move_constructible_v<T>;
static_assert(kStatusOrMoveCtorTracksT<int>,
              "StatusOr move-ctor noexcept must track T's move-ctor noexcept");
static_assert(kStatusOrMoveCtorTracksT<std::string>,
              "StatusOr move-ctor noexcept must track T's move-ctor noexcept");

// Compile-time contract: StatusOr<T>'s move-assignment noexcept tracks
// T's move-assignment noexcept (via std::variant). The contract must not be
// stronger than what T provides, or a throwing T move-assignment would call
// std::terminate() instead of propagating the exception.
template<typename T>
constexpr bool kStatusOrMoveAssignTracksT =
        std::is_nothrow_assignable_v<StatusOr<T>&, StatusOr<T>&&> ==
        std::is_nothrow_assignable_v<T&, T&&>;
static_assert(kStatusOrMoveAssignTracksT<int>,
              "StatusOr move-assign noexcept must track T's move-assignment noexcept");
static_assert(kStatusOrMoveAssignTracksT<std::string>,
              "StatusOr move-assign noexcept must track T's move-assignment noexcept");

namespace detail {

inline Status ExtractStatus(Status&& status) noexcept {
    return std::move(status);
}

template<typename T>
Status ExtractStatus(StatusOr<T>&& result) noexcept {
    return std::move(result).status();
}

}// namespace detail

}// namespace aethermind

#endif
