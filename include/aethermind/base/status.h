#ifndef AETHERMIND_AETHERMIND_BASE_STATUS_H
#define AETHERMIND_AETHERMIND_BASE_STATUS_H

#include "c_api.h"
#include "macros.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace aethermind {

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
    kUnauthenticated
};

AM_NODISCARD inline constexpr std::string_view StatusCodeName(StatusCode code) {
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
    }
    return "UNKNOWN";
}

AM_NODISCARD inline constexpr am_status_code ToAMStatusCode(StatusCode code) {
    return static_cast<am_status_code>(code);
}

AM_NODISCARD inline constexpr StatusCode FromAMStatusCode(am_status_code code) {
    return static_cast<StatusCode>(code);
}

class Status {
public:
    Status() : code_(StatusCode::kOk) {}

    explicit Status(StatusCode code, std::string message = "")
        : code_(code), message_(std::move(message)) {
        if (code_ == StatusCode::kOk) {
            message_.clear();
        }
    }

    AM_NODISCARD static Status Ok() {
        return Status();
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

    AM_NODISCARD bool ok() const {
        return code_ == StatusCode::kOk;
    }

    AM_NODISCARD explicit operator bool() const {
        return ok();
    }

    AM_NODISCARD StatusCode code() const {
        return code_;
    }

    AM_NODISCARD const std::string& message() const {
        return message_;
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

private:
    StatusCode code_;
    std::string message_;
};

template<typename T>
class StatusOr {
public:
    static_assert(!std::is_reference_v<T>, "StatusOr<T&> is not supported");
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>, "StatusOr<Status> is not supported");

    StatusOr(const T& value) : storage_(value) {}

    StatusOr(T&& value) : storage_(std::move(value)) {}

    StatusOr(const Status& status) : storage_(status) {
        if (status.ok()) {
            throw std::invalid_argument("StatusOr error constructor requires non-OK status");
        }
    }

    StatusOr(Status&& status) : storage_(std::move(status)) {
        if (std::get<Status>(storage_).ok()) {
            throw std::invalid_argument("StatusOr error constructor requires non-OK status");
        }
    }

    StatusOr(const StatusOr&) = default;
    StatusOr(StatusOr&&) noexcept = default;
    StatusOr& operator=(const StatusOr&) = default;
    StatusOr& operator=(StatusOr&&) noexcept = default;
    ~StatusOr() = default;

    AM_NODISCARD bool ok() const {
        return std::holds_alternative<T>(storage_);
    }

    AM_NODISCARD const T* get_if_ok() const noexcept {
        return ok() ? &std::get<T>(storage_) : nullptr;
    }

    AM_NODISCARD T* get_if_ok() noexcept {
        return ok() ? &std::get<T>(storage_) : nullptr;
    }

    AM_NODISCARD const Status& status() const& {
        static const Status kOkStatus = Status::Ok();
        if (ok()) {
            return kOkStatus;
        }
        return std::get<Status>(storage_);
    }

    AM_NODISCARD Status status() && {
        if (ok()) {
            return Status::Ok();
        }
        return std::move(std::get<Status>(storage_));
    }

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

    template<typename U>
    AM_NODISCARD T value_or(U&& fallback) const& {
        if (ok()) {
            return std::get<T>(storage_);
        }
        return static_cast<T>(std::forward<U>(fallback));
    }

    template<typename U>
    AM_NODISCARD T value_or(U&& fallback) && {
        if (ok()) {
            return std::move(std::get<T>(storage_));
        }
        return static_cast<T>(std::forward<U>(fallback));
    }

    AM_NODISCARD const T& operator*() const& {
        return value();
    }

    AM_NODISCARD T& operator*() & {
        return value();
    }

    AM_NODISCARD const T* operator->() const {
        return &value();
    }

    AM_NODISCARD T* operator->() {
        return &value();
    }

private:
    std::variant<T, Status> storage_;
};

}// namespace aethermind

#endif
