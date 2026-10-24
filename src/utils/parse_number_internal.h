#ifndef AETHERMIND_UTILS_PARSE_NUMBER_INTERNAL_H
#define AETHERMIND_UTILS_PARSE_NUMBER_INTERNAL_H

#include <string_view>

namespace aethermind::utils::detail {

#if defined(__APPLE__) || defined(__linux__)
// The production POSIX fallback is also compiled on Linux so its lexical,
// range, and allocation-failure contracts can be exercised directly.
bool ParseFloatFallback(std::string_view text, float& out) noexcept;
bool ParseDoubleFallback(std::string_view text, double& out) noexcept;
#endif

} // namespace aethermind::utils::detail

#endif // AETHERMIND_UTILS_PARSE_NUMBER_INTERNAL_H
