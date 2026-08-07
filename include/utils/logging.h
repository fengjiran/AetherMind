#ifndef AETHERMIND_UTILS_LOGGING_H
#define AETHERMIND_UTILS_LOGGING_H

/// @file
/// @brief Assertion macros and abort helpers for AetherMind.
///
/// `AM_CHECK` and `AM_DCHECK` express runtime and debug-only invariants.
/// Failed checks write a stable `Check failed: ...` prefix to standard error
/// before terminating the process. Death tests depend on that prefix.

#include <format>
#include <iostream>
#include <source_location>
#include <spdlog/spdlog.h>
#include <string_view>

namespace aethermind {

/// @brief Reports a failed check at a captured source location and aborts.
/// @param condition Text of the failed condition.
/// @param loc Source location of the check expression.
/// @note This helper is intended for use through `AM_CHECK`, which captures the
///       call-site location automatically.
inline void HandleCheckFailed(std::string_view condition,
                              std::source_location loc) {
    std::cerr << std::format("Check failed: ({}) at {}:{}:{}\n",
                             condition, loc.file_name(), loc.line(), loc.column());
    std::abort();
}

/// @brief Reports a failed check with a formatted message and aborts.
/// @tparam Args Types of the format arguments.
/// @param condition Text of the failed condition.
/// @param loc Source location of the check expression.
/// @param fmt Format string passed to `std::format`.
/// @param args Arguments consumed by `fmt`.
/// @throws std::format_error if formatting fails before the process is aborted.
/// @note This helper is intended for use through `AM_CHECK`.
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

/// @brief Evaluates an invariant and aborts when it is false.
/// @param condition Expression evaluated exactly once.
/// @param ... Optional `std::format` string and arguments for the failure
///             message.
/// @note This macro is active in every build. The `Check failed` output prefix
///       is part of the death-test contract.
#define AM_CHECK(condition, ...)                                                                       \
    do {                                                                                               \
        if (!(condition)) [[unlikely]] {                                                               \
            HandleCheckFailed(#condition, std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
        }                                                                                              \
    } while (false)

/// @brief Evaluates a debug-only invariant and aborts when it is false.
/// @param condition Expression evaluated once in debug builds.
/// @param ... Optional format string and arguments, evaluated only in debug
///             builds.
/// @note In `NDEBUG` builds, neither the condition nor the trailing arguments
///       are evaluated. Do not place required side effects in either position.
/// @note The macro preserves a dangling `else` so callers may attach a branch
///       that exists only in debug builds.
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
