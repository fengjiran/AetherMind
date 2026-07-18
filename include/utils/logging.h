// Assertion macros and abort helpers for AetherMind.
//
// AM_CHECK / AM_DCHECK express runtime and debug-only invariants. On
// failure they print a "Check failed: ..." line to stderr (with file,
// line, column, and an optional formatted message) and call std::abort().
// The "Check failed" prefix is a stable contract: death tests match it
// to detect the abort, so do not reword it.

#ifndef AETHERMIND_UTILS_LOGGING_H
#define AETHERMIND_UTILS_LOGGING_H

#include <format>
#include <iostream>
#include <source_location>
#include <spdlog/spdlog.h>
#include <string_view>

namespace aethermind {

// Abort helper invoked by AM_CHECK when `condition` fails. Not intended
// for direct use; call AM_CHECK so the source location is captured
// automatically. Writes the failure line to stderr and aborts.
inline void HandleCheckFailed(std::string_view condition,
                              std::source_location loc) {
    std::cerr << std::format("Check failed: ({}) at {}:{}:{}\n",
                             condition, loc.file_name(), loc.line(), loc.column());
    std::abort();
}

// Abort helper variant that appends a formatted message to the failure
// line. Same abort semantics as the overload above; `fmt` and `args`
// are forwarded to std::format.
template<typename... Args>
void HandleCheckFailed(std::string_view condition,
                       std::source_location loc,
                       std::format_string<Args...> fmt,
                       Args&&... args) {
    std::string message = std::format(fmt, std::forward<Args>(args)...);
    std::cerr << std::format("Check failed: ({}) at {}:{}:{} [{}]\nMessage: {}\n",
                             condition, loc.file_name(), loc.line(), loc.column(),
                             loc.function_name(), message);
    std::abort();
}

// Evaluates `condition` exactly once; on false, writes the failure line
// to stderr and aborts. Always live in every build — use AM_DCHECK for
// debug-only checks.
//
// Trailing variadic args, if present, are forwarded as a std::format
// message:
//   AM_CHECK(i < size, "index {} out of range {}", i, size);
//
// The "Check failed" prefix in the output is matched by death tests.
#define AM_CHECK(condition, ...)                                                                       \
    do {                                                                                               \
        if (!(condition)) [[unlikely]] {                                                               \
            HandleCheckFailed(#condition, std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
        }                                                                                              \
    } while (false)

// Debug-only variant of AM_CHECK: equivalent to AM_CHECK in debug builds,
// a no-op in release builds.
//
// Release-build hazard: in NDEBUG builds neither `condition` nor the
// trailing format args are evaluated — do not place side-effectful
// expressions in either position; they will silently stop running in
// release builds.
//
// The `while (false) if (...) ... else` skeleton (rather than a plain
// `while (false);`) leaves a dangling else so a caller may optionally
// attach an else-branch that runs only in debug builds:
//   AM_DCHECK(x > 0) else { /* debug-only fallback */ }
#ifdef NDEBUG
#define AM_RELEASE
#define AM_DCHECK(condition, ...)                     \
    while (false)                                     \
        if (static_cast<bool>(condition)) [[likely]]; \
        else
#else
#define AM_DEBUG
#define AM_DCHECK(condition, ...) AM_CHECK(condition __VA_OPT__(, ) __VA_ARGS__)
#endif

}// namespace aethermind

#endif// AETHERMIND_UTILS_LOGGING_H