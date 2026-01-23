//
// Created by richard on 1/24/26.
//
module;

#include <cstdlib>
#include <format>
#include <iostream>
#include <source_location>
#include <string_view>

export module check_module;

namespace aethermind {

export inline void HandleCheckFailed(std::string_view condition,
                                     std::source_location loc) {
    std::cerr << std::format("Check failed: ({}) at {}:{}:{}\n",
                             condition, loc.file_name(), loc.line(), loc.column());
    std::abort();
}

export template<typename... Args>
void HandleCheckFailedWithMsg(std::string_view condition,
                              std::source_location loc,
                              std::format_string<Args...> fmt,
                              Args&&... args) {
    std::string message = std::format(fmt, std::forward<Args>(args)...);
    std::cerr << std::format("Check failed: ({}) at {}:{}:{} [{}]\nMessage: {}\n",
                             condition, loc.file_name(), loc.line(), loc.column(),
                             loc.function_name(), message);
    std::abort();
}

}// namespace aethermind
