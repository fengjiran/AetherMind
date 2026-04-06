/// \file
/// Integer overflow checking utilities.
///
/// Provides checked arithmetic operations that detect overflow before it occurs.
/// Uses compiler builtins when available (GCC/Clang), otherwise falls back to
/// portable implementations.

#ifndef AETHERMIND_UTILS_OVERFLOW_CHECK_H
#define AETHERMIND_UTILS_OVERFLOW_CHECK_H

#include "utils/logging.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace aethermind {

#if defined(__GNUC__) || defined(__clang__)

/// Performs checked multiplication, detecting overflow.
/// \param a First operand.
/// \param b Second operand.
/// \param out Pointer to store the result (may be truncated on overflow).
/// \return true if overflow occurred, false otherwise.
/// \pre out != nullptr
template<typename T>
    requires std::is_integral_v<T>
inline bool mul_overflow(T a, T b, T* out) noexcept {
    return __builtin_mul_overflow(a, b, out);
}

/// Performs checked addition, detecting overflow.
/// \param a First operand.
/// \param b Second operand.
/// \param out Pointer to store the result (may be truncated on overflow).
/// \return true if overflow occurred, false otherwise.
/// \pre out != nullptr
template<typename T>
    requires std::is_integral_v<T>
inline bool add_overflow(T a, T b, T* out) noexcept {
    return __builtin_add_overflow(a, b, out);
}

/// Performs checked subtraction, detecting overflow.
/// \param a First operand.
/// \param b Second operand.
/// \param out Pointer to store the result (may be truncated on overflow).
/// \return true if overflow occurred, false otherwise.
/// \pre out != nullptr
template<typename T>
    requires std::is_integral_v<T>
inline bool sub_overflow(T a, T b, T* out) noexcept {
    return __builtin_sub_overflow(a, b, out);
}

#else

// Portable fallback implementations.

template<typename T>
    requires std::is_integral_v<T>
inline bool mul_overflow(T a, T b, T* out) noexcept {
    using U = std::make_unsigned_t<T>;
    const U result = static_cast<U>(a) * static_cast<U>(b);
    *out = static_cast<T>(result);

    if (a == 0 || b == 0) {
        return false;
    }
    return static_cast<T>(result / static_cast<U>(a)) != b;
}

template<typename T>
    requires std::is_integral_v<T>
inline bool add_overflow(T a, T b, T* out) noexcept {
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
    requires std::is_integral_v<T>
inline bool sub_overflow(T a, T b, T* out) noexcept {
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

/// Multiplies a range of non-negative integers into `*out`.
///
/// Iteratively multiplies all elements in [first, last), checking for overflow
/// at each step.
///
/// \tparam Iter Input iterator type.
/// \param first Start of range.
/// \param last End of range.
/// \param out Pointer to store the product.
/// \return true if overflow occurred, false otherwise.
///
/// \pre out != nullptr
/// \pre All elements must be non-negative and convertible to uint64_t.
///
/// \note On overflow, `*out` contains the last intermediate value and must not
/// be used as a valid result.
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

/// Multiplies all elements in a container into `*out`.
/// \overload
template<typename Container>
bool safe_multiply_u64(const Container& c, uint64_t* out) noexcept {
    return safe_multiply_u64(c.begin(), c.end(), out);
}

}// namespace aethermind


#endif// AETHERMIND_UTILS_OVERFLOW_CHECK_H