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

// Returns true if a and b are not both negative
template<typename T, typename U>
constexpr bool signs_differ(const T& a, const U& b) {
    return is_negative(a) != is_negative(b);
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

// Returns true if x is greater than the greatest value of the type Limit
template<typename Limit, typename T>
constexpr bool greater_than_max(const T& x) {
    constexpr bool can_overflow = std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
    return can_overflow && x > std::numeric_limits<Limit>::max();
}

// Returns true if x < lowest(Limit). Standard comparison
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& x,
        std::false_type /*limit_is_unsigned*/,
        std::false_type /*x_is_unsigned*/) {
    return x < std::numeric_limits<Limit>::lowest();
}

/// Returns false since all the limit is signed and therefore includes
/// negative values but x cannot be negative because it is unsigned
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& /*x*/,
        std::false_type /*limit_is_unsigned*/,
        std::true_type /*x_is_unsigned*/) {
    return false;
}

/// Returns true if x < 0, where 0 is constructed from T.
/// Limit is not signed, so its lower value is zero
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& x,
        std::true_type /*limit_is_unsigned*/,
        std::false_type /*x_is_unsigned*/) {
    return x < T(0);
}

/// Returns false sign both types are unsigned
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& /*x*/,
        std::true_type /*limit_is_unsigned*/,
        std::true_type /*x_is_unsigned*/) {
    return false;
}

/// Returns true if x is less than the lowest value of type T
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :
template<typename Limit, typename T>
constexpr bool less_than_lowest(const T& x) {
    return less_than_lowest<Limit>(x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

// bool can be converted to any type
template<typename From, typename To, typename R = std::enable_if_t<std::is_same_v<From, bool>, bool>>
R overflows(From, MAYBE_UNUSED bool strict_unsigned = false) {
    return false;
}

template<typename From, typename To, typename = void,
         typename R = std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>, bool>>
R overflows(From f, bool strict_unsigned = false) {
    using Limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if constexpr (!Limit::is_signed && std::numeric_limits<From>::is_signed) {
        if (!strict_unsigned) {
            return greater_than_max<To>(f) ||
                   (is_negative(f) && -static_cast<uint64_t>(f) > static_cast<uint64_t>(Limit::max()));
        }
    }

    return greater_than_max<To>(f) || less_than_lowest<To>(f);
}

}// namespace aethermind

#endif//AETHERMIND_CAST_H
