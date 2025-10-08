//
// Created by richard on 10/8/25.
//

#ifndef AETHERMIND_CAST_H
#define AETHERMIND_CAST_H

#include "macros.h"
#include "utils/complex.h"

namespace aethermind {

// Returns false since we cannot have x < 0 if x is unsigned.
template<typename T>
constexpr bool is_negative(const T&, std::true_type) {
    return false;
}

// Returns true if a signed variable x < 0
template<typename T>
constexpr bool is_negative(const T& x, std::false_type) {
    return x < T(0);
}

/// Returns true if x < 0
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, Half does not :-(
template<typename T>
constexpr bool is_negative(const T& x) {
    return is_negative(x, std::is_unsigned<T>());
}

// Returns the sign of an unsigned variable x as 0, 1
template<typename T>
constexpr int signum(const T& x, std::true_type) {
    return T(0) < x;
}

// Returns the sign of a signed variable x as -1, 0, 1
template<typename T>
constexpr int signum(const T& x, std::false_type) {
    return (T(0) < x) - (x < T(0));
}

/// Returns the sign of x as -1, 0, 1
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, Half does not :-(
template<typename T>
constexpr int signum(const T& x) {
    return signum(x, std::is_unsigned<T>());
}

// bool can be converted to any type
template<typename From, typename To, typename R = std::enable_if_t<std::is_same_v<From, bool>, bool>>
R overflows(From, MAYBE_UNUSED bool strict_unsigned = false) {
    return false;
}


}// namespace aethermind

#endif//AETHERMIND_CAST_H
