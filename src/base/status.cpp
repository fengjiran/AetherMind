#include "aethermind/base/status.h"

#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind {

Status Status::Cancelled(std::string_view message) {
    return Status(StatusCode::kCancelled, std::string(message));
}

Status Status::Unknown(std::string_view message) {
    return Status(StatusCode::kUnknown, std::string(message));
}

Status Status::DeadlineExceeded(std::string_view message) {
    return Status(StatusCode::kDeadlineExceeded, std::string(message));
}

Status Status::Aborted(std::string_view message) {
    return Status(StatusCode::kAborted, std::string(message));
}

Status Status::Unavailable(std::string_view message) {
    return Status(StatusCode::kUnavailable, std::string(message));
}

Status Status::DataLoss(std::string_view message) {
    return Status(StatusCode::kDataLoss, std::string(message));
}

Status Status::Unauthenticated(std::string_view message) {
    return Status(StatusCode::kUnauthenticated, std::string(message));
}

Status Status::InvalidArgument(std::string_view message) {
    return Status(StatusCode::kInvalidArgument, std::string(message));
}

Status Status::NotFound(std::string_view message) {
    return Status(StatusCode::kNotFound, std::string(message));
}

Status Status::Internal(std::string_view message) {
    return Status(StatusCode::kInternal, std::string(message));
}

Status Status::PermissionDenied(std::string_view message) {
    return Status(StatusCode::kPermissionDenied, std::string(message));
}

Status Status::AlreadyExists(std::string_view message) {
    return Status(StatusCode::kAlreadyExists, std::string(message));
}

Status Status::OutOfRange(std::string_view message) {
    return Status(StatusCode::kOutOfRange, std::string(message));
}

Status Status::FailedPrecondition(std::string_view message) {
    return Status(StatusCode::kFailedPrecondition, std::string(message));
}

Status Status::Unimplemented(std::string_view message) {
    return Status(StatusCode::kUnimplemented, std::string(message));
}

Status Status::ResourceExhausted(std::string_view message) {
    return Status(StatusCode::kResourceExhausted, std::string(message));
}

Status Status::Overflow(std::string_view message) {
    return Status(StatusCode::kOverflow, std::string(message));
}

Status Status::WithMessage(std::string message) const noexcept {
    if (ok()) {
        return Ok();
    }
    return Status(code_, std::move(message));
}

std::string Status::ToString() const {
    if (ok()) {
        return std::string(StatusCodeName(StatusCode::kOk));
    }

    if (message_.empty()) {
        return std::string(StatusCodeName(code_));
    }

    return std::string(StatusCodeName(code_)) + ": " + message_;
}

std::ostream& operator<<(std::ostream& os, const Status& status) {
    return os << status.ToString();
}

} // namespace aethermind
