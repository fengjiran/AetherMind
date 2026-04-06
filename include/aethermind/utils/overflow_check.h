//
// Created by richard on 4/2/26.
//

#ifndef AETHERMIND_UTILS_OVERFLOW_CHECK_H
#define AETHERMIND_UTILS_OVERFLOW_CHECK_H

#include "utils/logging.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace aethermind {

#if defined(__GNUC__) || defined(__clang__)
// Use compiler builtin for best performance
template<typename T>
inline bool mul_overflow(T a, T b, T* out) noexcept {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    return __builtin_mul_overflow(a, b, out);
}

template<typename T>
inline bool add_overflow(T a, T b, T* out) noexcept {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    return __builtin_add_overflow(a, b, out);
}

template<typename T>
inline bool sub_overflow(T a, T b, T* out) noexcept {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    return __builtin_sub_overflow(a, b, out);
}
#else
// Portable fallback
template<typename T>
inline bool mul_overflow(T a, T b, T* out) noexcept {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    using U = std::make_unsigned_t<T>;
    const U result = static_cast<U>(a) * static_cast<U>(b);
    *out = static_cast<T>(result);

    if (a == 0 || b == 0) {
        return false;
    }
    return static_cast<T>(result / static_cast<U>(a)) != b;
}

template<typename T>
inline bool add_overflow(T a, T b, T* out) noexcept {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    if constexpr (std::is_unsigned_v<T>) {
        const T result = a + b;
        *out = result;
        return result < a;
    } else {
        using U = std::make_unsigned_t<T>;
        const U result = static_cast<U>(a) + static_cast<U>(b);
        *out = static_cast<T>(result);

        // Positive overflow: both positive, result negative
        // Negative overflow: both negative, result positive
        return (a > 0 && b > 0 && static_cast<T>(result) < 0) ||
               (a < 0 && b < 0 && static_cast<T>(result) > 0);
    }
}

template<typename T>
inline bool sub_overflow(T a, T b, T* out) noexcept {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    if constexpr (std::is_unsigned_v<T>) {
        const T result = a - b;
        *out = result;
        return result > a;
    } else {
        using U = std::make_unsigned_t<T>;
        const U result = static_cast<U>(a) - static_cast<U>(b);
        *out = static_cast<T>(result);

        // Overflow: a >= 0, b < 0, result < 0
        // Underflow: a < 0, b > 0, result > 0
        return (a >= 0 && b < 0 && static_cast<T>(result) < 0) ||
               (a < 0 && b > 0 && static_cast<T>(result) > 0);
    }
}
#endif

/// Multiplies a range of non-negative integer values into `*out`.
///
/// Requirements:
/// - `out` must not be null.
/// - every element in [first, last) must be non-negative and convertible to uint64_t.
///
/// Returns:
/// - true if any intermediate multiplication overflowed uint64_t.
/// - false otherwise.
///
/// Note:
/// - when true is returned, `*out` contains the last intermediate value and must
///   not be treated as a valid final product.
template<typename Iter>
bool safe_multiply_u64(Iter first, Iter last, uint64_t* out) noexcept {
    AM_DCHECK(out != nullptr);

    uint64_t prod = 1;
    bool overflowed = false;
    while (first != last) {
        using ValueType = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
        if constexpr (std::is_signed_v<ValueType>) {
            AM_DCHECK(*first >= 0);
        }
        overflowed |= mul_overflow(prod, static_cast<uint64_t>(*first), &prod);
        ++first;
    }
    *out = prod;
    return overflowed;
}

template<typename Container>
bool safe_multiply_u64(const Container& c, uint64_t* out) noexcept {
    return safe_multiply_u64(c.begin(), c.end(), out);
}

}// namespace aethermind


#endif// AETHERMIND_UTILS_OVERFLOW_CHECK_H