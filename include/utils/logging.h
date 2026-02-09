//
// Created by richard on 1/24/26.
//

#ifndef AETHERMIND_UTILS_LOGGING_H
#define AETHERMIND_UTILS_LOGGING_H

#include <format>
#include <iostream>
#include <source_location>
#include <spdlog/spdlog.h>
#include <string_view>

namespace aethermind {

inline void HandleCheckFailed(std::string_view condition,
                              std::source_location loc) {
    std::cerr << std::format("Check failed: ({}) at {}:{}:{}\n",
                             condition, loc.file_name(), loc.line(), loc.column());
    std::abort();
}

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

#define AM_CHECK(condition, ...)                                                                       \
    do {                                                                                               \
        if (!(condition)) [[unlikely]] {                                                               \
            HandleCheckFailed(#condition, std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
        }                                                                                              \
    } while (false)

#ifdef NDEBUG
#define AM_DCHECK(condition, ...)                     \
    while (false)                                     \
        if (static_cast<bool>(condition)) [[likely]]; \
        else
#else
#define AM_DCHECK(condition, ...) AM_CHECK(condition __VA_OPT__(, ) __VA_ARGS__)
#endif


}// namespace aethermind

#endif//AETHERMIND_UTILS_LOGGING_H
