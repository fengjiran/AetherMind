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
/// Maps to am_status_code for C ABI compatibility via ToAMStatusCode()/FromAMStatusCode().
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
    return "UNKNOWN";
}

AM_NODISCARD constexpr am_status_code ToAMStatusCode(StatusCode code) noexcept {
    return static_cast<am_status_code>(code);
}

AM_NODISCARD constexpr StatusCode FromAMStatusCode(am_status_code code) noexcept {
    if (code < AM_STATUS_OK || code > AM_STATUS_UNAUTHENTICATED) {
        return StatusCode::kUnknown;
    }
    return static_cast<StatusCode>(code);
}

/// Represents the result of an operation that may fail.
///
/// Modeled after absl::Status and gRPC status. Default-constructed Status is OK.
/// Thread-compatible (const methods are thread-safe).
///
/// @note Use StatusOr<T> for operations that return a value on success.
class Status {
public:
    /// Creates an OK status.
    Status() noexcept : code_(StatusCode::kOk) {}

    AM_NODISCARD static Status Ok() noexcept {
        return {};
    }

    AM_NODISCARD static Status InvalidArgument(std::string_view message) {
        return Status(StatusCode::kInvalidArgument, std::string(message));
    }

    AM_NODISCARD static Status NotFound(std::string_view message) {
        return Status(StatusCode::kNotFound, std::string(message));
    }

    AM_NODISCARD static Status Internal(std::string_view message) {
        return Status(StatusCode::kInternal, std::string(message));
    }

    AM_NODISCARD static Status PermissionDenied(std::string_view message) {
        return Status(StatusCode::kPermissionDenied, std::string(message));
    }

    AM_NODISCARD static Status AlreadyExists(std::string_view message) {
        return Status(StatusCode::kAlreadyExists, std::string(message));
    }

    AM_NODISCARD static Status OutOfRange(std::string_view message) {
        return Status(StatusCode::kOutOfRange, std::string(message));
    }

    AM_NODISCARD static Status FailedPrecondition(std::string_view message) {
        return Status(StatusCode::kFailedPrecondition, std::string(message));
    }

    AM_NODISCARD static Status Unimplemented(std::string_view message) {
        return Status(StatusCode::kUnimplemented, std::string(message));
    }

    AM_NODISCARD static Status ResourceExhausted(std::string_view message) {
        return Status(StatusCode::kResourceExhausted, std::string(message));
    }

    AM_NODISCARD static Status Overflow(std::string_view message) {
        return Status(StatusCode::kOverflow, std::string(message));
    }

    AM_NODISCARD bool ok() const noexcept {
        return code_ == StatusCode::kOk;
    }

    AM_NODISCARD explicit operator bool() const noexcept {
        return ok();
    }

    AM_NODISCARD StatusCode code() const noexcept {
        return code_;
    }

    AM_NODISCARD const std::string& message() const noexcept {
        return message_;
    }

    AM_NODISCARD Status WithMessage(std::string message) const noexcept {
        if (ok()) {
            return Ok();
        }
        return Status(code_, std::move(message));
    }

    AM_NODISCARD std::string ToString() const {
        if (ok()) {
            return std::string(StatusCodeName(StatusCode::kOk));
        }

        if (message_.empty()) {
            return std::string(StatusCodeName(code_));
        }

        return std::string(StatusCodeName(code_)) + ": " + message_;
    }

    AM_NODISCARD bool operator==(const Status& other) const {
        return code_ == other.code_ && message_ == other.message_;
    }

    AM_NODISCARD bool operator!=(const Status& other) const {
        return !(*this == other);
    }

    AM_NODISCARD bool operator==(StatusCode code) const noexcept {
        return code_ == code;
    }

    AM_NODISCARD bool operator!=(StatusCode code) const noexcept {
        return code_ != code;
    }

private:
    explicit Status(StatusCode code, std::string message) noexcept
        : code_(code), message_(std::move(message)) {}

    StatusCode code_;
    std::string message_;
};

/// Evaluates expr and returns its status if not OK.
///
/// Compatible with both Status and StatusOr<T> return types.
/// The expression is evaluated exactly once.
///
/// Example:
///   StatusOr<int> ParseInt(const std::string& s);
///   Status Run() {
///     AM_RETURN_IF_ERROR(ParseInt("42"));  // Returns StatusOr<int>, converted to Status on error
///     return Status::Ok();
///   }
#define AM_RETURN_IF_ERROR(expr)                   \
    do {                                           \
        auto am_status_result_ = (expr);           \
        if (!am_status_result_.ok()) AM_UNLIKELY { \
                return am_status_result_;          \
            }                                      \
    } while (false)

/// Evaluates expr and returns an augmented error status if not OK.
///
/// @warning Only use in functions returning Status. For StatusOr<T> return types,
///          this macro will fail to compile because Status cannot implicitly convert
///          to StatusOr<T>. Use AM_RETURN_IF_ERROR and construct error separately.
///
/// @param expr Expression returning Status or StatusOr<T>.
/// @param msg  Prefix message to prepend to the original error message.
///
/// Example:
///   Status LoadModel(const std::string& path) {
///     AM_RETURN_IF_ERROR_WITH_MSG(OpenFile(path), "Failed to load model");
///     // On error, returns: Status(INTERNAL, "Failed to load model: file not found")
///     return Status::Ok();
///   }
#define AM_RETURN_IF_ERROR_WITH_MSG(expr, msg)                                  \
    do {                                                                        \
        auto am_status_result_ = (expr);                                        \
        if (!am_status_result_.ok()) AM_UNLIKELY {                              \
                return am_status_result_.WithMessage(                           \
                        std::string(msg) + ": " + am_status_result_.message()); \
            }                                                                   \
    } while (false)

#define AM_ASSIGN_OR_RETURN_CONCAT_(a, b) a##b
#define AM_ASSIGN_OR_RETURN_CONCAT(a, b) AM_ASSIGN_OR_RETURN_CONCAT_(a, b)

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
#define AM_ASSIGN_OR_RETURN(lhs, expr)                                     \
    AM_ASSIGN_OR_RETURN_IMPL(                                              \
            AM_ASSIGN_OR_RETURN_CONCAT(am_assign_or_result_, __COUNTER__), \
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
/// @tparam T The value type returned on success. Must be move-constructible.
///
/// Example:
///   StatusOr<int> ParseInt(const std::string& s);
///   auto result = ParseInt("42");
///   if (result.ok()) { use(*result); } else { handle(result.status()); }
template<typename T>
class StatusOr {
public:
    /// Static assertions ensure T is a valid type for StatusOr.
    static_assert(!std::is_reference_v<T>, "StatusOr<T&> is not supported");
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>, "StatusOr<Status> is not supported");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "StatusOr<T> requires T to be nothrow move-constructible for noexcept guarantee");

    template<typename U = T>
        requires(!std::same_as<std::decay_t<U>, Status>) &&
                (!std::same_as<std::decay_t<U>, StatusOr>) &&
                std::convertible_to<U, T>
    StatusOr(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>)
        : storage_(std::in_place_type<T>, std::forward<U>(value)) {}// NOLINT

    // inplace ctor
    template<typename... Args>
    StatusOr(std::in_place_t, Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)// NOLINT
        : storage_(std::in_place_type<T>, std::forward<Args>(args)...) {}

    /// Constructs from an error status. Throws std::invalid_argument if status.ok().
    ///
    /// @throws std::invalid_argument if status is OK (prevents accidentally wrapping success as error).
    StatusOr(const Status& status) : storage_(status) {
        if (status.ok()) {
            throw std::invalid_argument("StatusOr error constructor requires non-OK status");
        }
    }

    StatusOr(Status&& status) : storage_(std::move(status)) {// NOLINT
        if (std::get<Status>(storage_).ok()) {
            throw std::invalid_argument("StatusOr error constructor requires non-OK status");
        }
    }

    StatusOr(const StatusOr&) = default;
    StatusOr(StatusOr&&) noexcept = default;
    StatusOr& operator=(const StatusOr&) = default;
    StatusOr& operator=(StatusOr&&) noexcept = default;
    ~StatusOr() = default;

    AM_NODISCARD bool ok() const noexcept {
        return std::holds_alternative<T>(storage_);
    }

    AM_NODISCARD const T* get_if_ok() const noexcept {
        return ok() ? &std::get<T>(storage_) : nullptr;
    }

    AM_NODISCARD T* get_if_ok() noexcept {
        return ok() ? &std::get<T>(storage_) : nullptr;
    }

    AM_NODISCARD const Status& status() const& noexcept {
        static const Status kOkStatus = Status::Ok();
        if (ok()) {
            return kOkStatus;
        }
        return std::get<Status>(storage_);
    }

    AM_NODISCARD Status status() && noexcept {
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
            throw std::logic_error("Attempted to access value() on error StatusOr: " + status().ToString());
        }
        return std::get<T>(storage_);
    }

    AM_NODISCARD T& value() & {
        if (!ok()) {
            throw std::logic_error("Attempted to access value() on error StatusOr: " + status().ToString());
        }
        return std::get<T>(storage_);
    }

    AM_NODISCARD T&& value() && {
        if (!ok()) {
            throw std::logic_error("Attempted to access value() on error StatusOr: " + status().ToString());
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

    AM_NODISCARD const T& operator*() const& noexcept {
        return std::get<T>(storage_);
    }

    AM_NODISCARD T& operator*() & noexcept {
        return std::get<T>(storage_);
    }

    AM_NODISCARD const T&& operator*() const&& noexcept {
        return std::move(std::get<T>(storage_));
    }

    AM_NODISCARD T&& operator*() && noexcept {
        return std::move(std::get<T>(storage_));
    }

    AM_NODISCARD const T* operator->() const& noexcept {
        return &std::get<T>(storage_);
    }

    AM_NODISCARD T* operator->() & noexcept {
        return &std::get<T>(storage_);
    }

    AM_NODISCARD const T* operator->() const&& noexcept {
        return &std::get<T>(storage_);
    }

    AM_NODISCARD T* operator->() && noexcept {
        return &std::get<T>(storage_);
    }

    AM_NODISCARD bool operator==(StatusCode code) const noexcept {
        return status().code() == code;
    }

    AM_NODISCARD bool operator!=(StatusCode code) const noexcept {
        return status().code() != code;
    }

private:
    std::variant<T, Status> storage_;
};

}// namespace aethermind

#endif
