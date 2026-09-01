#ifndef AETHERMIND_UTILS_VARIANT_UTILS_H
#define AETHERMIND_UTILS_VARIANT_UTILS_H

/// @file
/// @brief Utilities for composing `std::variant` visitor overloads.

namespace aethermind {

/// @brief Combines call operators from multiple callable types into one visitor.
///
/// The resulting object is intended for use with `std::visit`.
/// @tparam Ts Callable types whose `operator()` members are exposed.
template<typename... Ts>
// NOLINTNEXTLINE(readability-identifier-naming)
struct overloaded : Ts... {
    using Ts::operator()...;
};

/// @brief Deduces an `overloaded` visitor from its callable arguments.
template<typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace aethermind

#endif // AETHERMIND_UTILS_VARIANT_UTILS_H
