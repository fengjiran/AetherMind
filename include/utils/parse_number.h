#ifndef AETHERMIND_UTILS_PARSE_NUMBER_H
#define AETHERMIND_UTILS_PARSE_NUMBER_H

/// @file
/// @brief Locale-independent floating-point literal parsing.
///
/// std::from_chars floating-point overloads are unavailable on Apple platforms
/// whose deployment target predates macOS 26.0 (libc++ marks them unavailable
/// below that version). These helpers route through std::from_chars where
/// supported and fall back to strtod_l/strtof_l with the "C" locale otherwise,
/// preserving locale-independent, full-consumption semantics everywhere.

#include <string_view>

namespace aethermind::utils {

/// @brief Parses `text` as a float literal, consuming the entire input.
/// @param text Candidate literal; must be non-empty.
/// @param out Receives the parsed value on success.
/// @return True if `text` is a complete, valid floating-point literal.
bool ParseFloat(std::string_view text, float& out) noexcept;

/// @brief Parses `text` as a double literal, consuming the entire input.
/// @param text Candidate literal; must be non-empty.
/// @param out Receives the parsed value on success.
/// @return True if `text` is a complete, valid floating-point literal.
bool ParseDouble(std::string_view text, double& out) noexcept;

} // namespace aethermind::utils

#endif // AETHERMIND_UTILS_PARSE_NUMBER_H