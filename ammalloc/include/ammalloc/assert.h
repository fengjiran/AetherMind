// Assertion macros and abort helpers for ammalloc.
//
// Self-contained counterpart of include/utils/logging.h: AMMALLOC_CHECK /
// AMMALLOC_DCHECK express runtime and debug-only invariants. On failure they
// print a "Check failed: ..." line to stderr (with file, line, column,
// and an optional formatted message) and call std::abort(). The "Check
// failed" prefix is a stable contract: death tests match it to detect
// the abort, so do not reword it.
//
// Kept in sync with include/utils/logging.h by intent — this file exists
// so that ammalloc's public headers do not depend on the AetherMind
// include tree and can be built/distributed standalone.

#ifndef AMMALLOC_ASSERT_H
#define AMMALLOC_ASSERT_H

#include <cstdlib>
#include <format>
#include <iostream>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace ammalloc::ammalloc_detail {

// Abort helper invoked by AMMALLOC_CHECK when `condition` fails. Not intended
// for direct use; call AMMALLOC_CHECK so the source location is captured
// automatically. Writes the failure line to stderr and aborts.
inline void HandleCheckFailed(std::string_view condition, std::source_location loc) {
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

}// namespace ammalloc::ammalloc_detail

// Evaluates `condition` exactly once; on false, writes the failure line
// to stderr and aborts. Always live in every build — use AMMALLOC_DCHECK for
// debug-only checks.
//
// Trailing variadic args, if present, are forwarded as a std::format
// message:
//   AMMALLOC_CHECK(i < size, "index {} out of range {}", i, size);
//
// The "Check failed" prefix in the output is matched by death tests.
#define AMMALLOC_CHECK(condition, ...)                                                       \
    do {                                                                                     \
        if (!(condition)) [[unlikely]] {                                                     \
            ::ammalloc::ammalloc_detail::HandleCheckFailed(                                  \
                    #condition, std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
        }                                                                                    \
    } while (false)

// Debug-only variant of AMMALLOC_CHECK: equivalent to AMMALLOC_CHECK in debug builds,
// a no-op in release builds.
//
// Release-build hazard: in NDEBUG builds neither `condition` nor the
// trailing format args are evaluated — do not place side-effectful
// expressions in either position; they will silently stop running in
// release builds.
#ifdef NDEBUG
#define AMMALLOC_DCHECK(condition, ...)               \
    while (false)                                     \
        if (static_cast<bool>(condition)) [[likely]]; \
        else
#else
#define AMMALLOC_DCHECK(condition, ...) AMMALLOC_CHECK(condition __VA_OPT__(, ) __VA_ARGS__)
#endif

#endif// AMMALLOC_ASSERT_H
