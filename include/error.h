/// Error exception type and throwing macros for AetherMind.
///
/// Defines `Error`, the base exception carrying a kind, message, and
/// traceback, and the `AM_THROW` / `AM_THROW_SILENT` macros that build and
/// throw `Error` via `ErrorBuilder`.

#ifndef AETHERMIND_ERROR_H
#define AETHERMIND_ERROR_H

#include "c_api.h"
#include "traceback.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {

/// Base exception type for all AetherMind errors.
///
/// Carries a kind (e.g., "value_error"), a human-readable message, and a
/// pre-captured traceback string. The `what()` method formats these into the
/// standard "Traceback (most recent call last):" layout using thread-local
/// storage for its return buffer.
class Error : public std::exception {
public:
    Error(std::string kind, std::string message, std::string traceback)
        : kind_(std::move(kind)), message_(std::move(message)), traceback_(std::move(traceback)) {}

    AM_NODISCARD const std::string& kind() const noexcept {
        return kind_;
    }

    AM_NODISCARD const std::string& message() const noexcept {
        return message_;
    }

    AM_NODISCARD const std::string& traceback() const noexcept {
        return traceback_;
    }

    /// Returns the full error message formatted as
    /// "Traceback (most recent call last):\n<reversed traceback><kind>: <message>".
    ///
    /// The returned pointer points to thread-local storage and is valid only
    /// until the next call to what() on any Error object within the same
    /// thread. Callers that need to retain the result should copy it into a
    /// local std::string. The result is cached per-thread per-Error: repeated
    /// calls on the same Error object return the cached string without
    /// recomputation.
    AM_NODISCARD const char* what() const noexcept override {
        thread_local std::string what_str;
        // Identity check: skip reformatting if this Error was the last one
        // formatted on the current thread.
        thread_local const Error* cached = nullptr;
        if (cached == this) {
            return what_str.c_str();
        }

        try {
            what_str = "Traceback (most recent call last):\n" +
                       TracebackMostRecent() + kind_ + ": " + message_;
        } catch (...) {
            // Fallback: skip traceback if formatting fails (e.g., bad_alloc).
            what_str = kind_ + ": " + message_;
        }
        cached = this;
        return what_str.c_str();
    }

    /// Replaces or appends to the traceback of an existing Error.
    ///
    /// @param e Target error. Must not be null. Ownership is not transferred.
    /// @param traceback New traceback content.
    /// @param update_mode kBacktraceUpdateModeReplace overwrites;
    ///        kBacktraceUpdateModeAppend concatenates.
    static void UpdateTraceback(Error* e, const std::string& traceback, BacktraceUpdateMode update_mode) {
        if (update_mode == kBacktraceUpdateModeReplace) {
            e->traceback_ = traceback;
        } else {
            e->traceback_ = e->traceback_ + traceback;
        }
    }

    /// Returns the traceback reversed so that the most recent call appears last.
    AM_NODISCARD std::string TracebackMostRecent() const {
        std::vector<size_t> newline_positions;
        for (size_t i = 0; i < traceback_.size(); ++i) {
            if (traceback_[i] == '\n') {
                newline_positions.push_back(i);
            }
        }

        std::string res;
        res.reserve(traceback_.size());

        // Walk newline positions in reverse to emit segments most-recent-last.
        size_t segment_end = traceback_.size();
        for (size_t i = newline_positions.size(); i > 0; --i) {
            size_t newline_pos = newline_positions[i - 1];
            size_t segment_start = newline_pos + 1;
            if (segment_start < segment_end) {
                res.append(traceback_.data() + segment_start, segment_end - segment_start);
                res.push_back('\n');
            }
            segment_end = newline_pos;
        }

        // Emit the segment before the first '\n' (if any).
        if (segment_end > 0) {
            res.append(traceback_.data(), segment_end);
            res.push_back('\n');
        }

        return res;
    }

private:
    std::string kind_;
    std::string message_;
    std::string traceback_;
};

/// RAII helper that accumulates a message via `<<` and throws an `Error`
/// when destroyed.
///
/// Used by the `AM_THROW` and `AM_THROW_SILENT` macros. The destructor
/// throws unless the program is already in stack unwinding, in which case
/// throwing would call `std::terminate`.
class ErrorBuilder {
public:
    ErrorBuilder(std::string kind, std::string traceback, bool log_before_throw)
        : kind_(std::move(kind)), traceback_(std::move(traceback)), log_before_throw_(log_before_throw) {}

    std::ostringstream& stream() {
        return stream_;
    }

    ~ErrorBuilder() AETHERMIND_THROW_EXCEPTION {
        // Suppress throw if already in stack unwinding to avoid std::terminate.
        if (std::uncaught_exceptions() > 0) {
            return;
        }

        Error error(kind_, stream_.str(), traceback_);
        if (log_before_throw_) {
            std::cerr << error.what() << "\n\n";
        }
        throw error;
    }

private:
    std::string kind_;
    std::string traceback_;
    std::ostringstream stream_;
    bool log_before_throw_;
};

/// Captures the current source location as a traceback string.
#define TRACEBACK_HERE AetherMindTraceback(__FILE__, __LINE__, FUNC_SIG, 0)

/// Builds and throws an Error of the given kind, logging the full message to
/// stderr before throwing. Usage: `AM_THROW(value_error) << "details"`.
#define AM_THROW(ErrorKind) ErrorBuilder(#ErrorKind, TRACEBACK_HERE, true).stream()

/// Like AM_THROW but suppresses stderr logging before throwing.
#define AM_THROW_SILENT(ErrorKind) ErrorBuilder(#ErrorKind, TRACEBACK_HERE, false).stream()

}// namespace aethermind

#endif// AETHERMIND_ERROR_H
