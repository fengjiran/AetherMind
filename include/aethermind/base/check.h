#ifndef AETHERMIND_BASE_CHECK_H
#define AETHERMIND_BASE_CHECK_H

/// @file check.h
/// @brief Assertion macros and abort helpers for AetherMind.
///
/// `AM_CHECK` and `AM_DCHECK` express runtime and debug-only invariants.
/// Failed checks write a stable `Check failed: ...` prefix to standard error
/// before terminating the process. Death tests depend on that prefix.

#include <cstdio>
#include <cstdlib>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind {

namespace detail {

[[noreturn]] inline void AbortFailedCheck(std::string_view condition, std::source_location loc,
                                          const std::string* message) noexcept {
    // This path must also work when building the diagnostic exhausted memory.
    std::fputs("Check failed: (", stderr);
    if (!condition.empty()) {
        std::fwrite(condition.data(), 1, condition.size(), stderr);
    }

    std::fprintf(stderr, ") at %s:%lu:%lu", loc.file_name(),
                 static_cast<unsigned long>(loc.line()), static_cast<unsigned long>(loc.column()));
    if (message) {
        std::fprintf(stderr, " [%s]\nMessage: ", loc.function_name());
        std::fwrite(message->data(), 1, message->size(), stderr);
    }

    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::abort();
}

} // namespace detail

/// @brief Reports a failed check at a captured source location and aborts.
/// @param condition Text of the failed condition.
/// @param loc Source location of the check expression.
/// @note This helper is intended for use through `AM_CHECK`, which captures the
///       call-site location automatically.
[[noreturn]] inline void HandleCheckFailed(std::string_view condition,
                                           std::source_location loc) noexcept {
    detail::AbortFailedCheck(condition, loc, nullptr);
}

/// @brief Reports a failed check with a formatted message and aborts.
/// @tparam Args Types of the format arguments.
/// @param condition Text of the failed condition.
/// @param loc Source location of the check expression.
/// @param fmt Format string passed to `std::format`.
/// @param args Arguments consumed by `fmt`.
/// @note If formatting throws, the diagnostic still includes the condition and location.
/// @note This helper is intended for use through `AM_CHECK`.
template<typename... Args>
[[noreturn]] void HandleCheckFailed(std::string_view condition,
                                    std::source_location loc,
                                    std::format_string<Args...> fmt,
                                    Args&&... args) noexcept {
    try {
        const auto message = std::format(fmt, std::forward<Args>(args)...);
        detail::AbortFailedCheck(condition, loc, &message);
    } catch (...) {
        HandleCheckFailed(condition, loc);
    }
}

/// @brief Evaluates an invariant and aborts when it is false.
/// @param condition Expression evaluated exactly once.
/// @param ... Optional `std::format` string and arguments for the failure
///             message.
/// @note This macro is active in every build. The `Check failed` output prefix
///       is part of the death-test contract.
/// @note Diagnostic arguments are evaluated only on failure. If evaluation
///       throws before the helper is entered, report the condition and abort.
#define AM_CHECK(condition, ...)                                                                                           \
    do {                                                                                                                   \
        if (!static_cast<bool>(condition)) [[unlikely]] {                                                                  \
            try {                                                                                                          \
                ::aethermind::HandleCheckFailed(#condition, ::std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
            } catch (...) {                                                                                                \
                ::aethermind::HandleCheckFailed(#condition, ::std::source_location::current());                            \
            }                                                                                                              \
        }                                                                                                                  \
    } while (false)

/// @brief Evaluates a debug-only invariant and aborts when it is false.
/// @param condition Expression evaluated once in debug builds.
/// @param ... Optional format string and arguments, evaluated only in debug
///             builds.
/// @note In `NDEBUG` builds, neither the condition nor the trailing arguments
///       are evaluated. Do not place required side effects in either position.
#ifdef NDEBUG
#define AM_RELEASE
#define AM_DCHECK(condition, ...) \
    do {                          \
    } while (false && static_cast<bool>(condition))
#else
#define AM_DEBUG
#define AM_DCHECK(condition, ...) AM_CHECK(condition __VA_OPT__(, ) __VA_ARGS__)
#endif

} // namespace aethermind

#endif // AETHERMIND_BASE_CHECK_H
