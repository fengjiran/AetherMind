#ifndef AETHERMIND_UTILS_OVERFLOW_CHECK_H
#define AETHERMIND_UTILS_OVERFLOW_CHECK_H

/// @file
/// @brief Checked integer arithmetic and range-product utilities.
///
/// Provides checked arithmetic operations that detect overflow before it occurs.
/// Uses compiler builtins when available (GCC/Clang), otherwise falls back to
/// portable implementations.

#include "utils/logging.h"

#include <cstdint>
#include <type_traits>

namespace aethermind {

/// @brief Multiplies two integral values and reports whether the result overflowed.
///
/// The result is always written to `out`. If overflow occurs, the stored value
/// is the representable result for `T` and must not be used as an exact product.
///
/// @tparam T Integral operand and result type.
/// @param a First operand.
/// @param b Second operand.
/// @param out Destination for the representable result.
/// @return True if the exact product is not representable in `T`; false otherwise.
/// @pre `out` is not null.
#if defined(__GNUC__) || defined(__clang__)
template<typename T>
    requires std::is_integral_v<T>
bool CheckOverflowMul(T a, T b, T* out) noexcept {
    return __builtin_mul_overflow(a, b, out);
}

/// @brief Adds two integral values and reports whether the result overflowed.
///
/// The result is always written to `out`. If overflow occurs, the stored value
/// is the representable result for `T` and must not be used as an exact sum.
///
/// @tparam T Integral operand and result type.
/// @param a First operand.
/// @param b Second operand.
/// @param out Destination for the representable result.
/// @return True if the exact sum is not representable in `T`; false otherwise.
/// @pre `out` is not null.
template<typename T>
    requires std::is_integral_v<T>
bool CheckOverflowAdd(T a, T b, T* out) noexcept {
    return __builtin_add_overflow(a, b, out);
}

/// @brief Subtracts two integral values and reports whether the result overflowed.
///
/// The result is always written to `out`. If overflow occurs, the stored value
/// is the representable result for `T` and must not be used as an exact difference.
///
/// @tparam T Integral operand and result type.
/// @param a First operand.
/// @param b Second operand.
/// @param out Destination for the representable result.
/// @return True if the exact difference is not representable in `T`; false otherwise.
/// @pre `out` is not null.
template<typename T>
    requires std::is_integral_v<T>
bool CheckOverflowSub(T a, T b, T* out) noexcept {
    return __builtin_sub_overflow(a, b, out);
}

#else

// Portable fallback implementations use unsigned arithmetic to avoid signed
// overflow before checking whether the converted result changed sign.

template<typename T>
    requires std::is_integral_v<T>
inline bool CheckOverflowMul(T a, T b, T* out) noexcept {
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
inline bool CheckOverflowAdd(T a, T b, T* out) noexcept {
    if constexpr (std::is_unsigned_v<T>) {
        const T result = a + b;
        *out = result;
        return result < a;
    } else {
        using U = std::make_unsigned_t<T>;
        const U result = static_cast<U>(a) + static_cast<U>(b);
        *out = static_cast<T>(result);

        // Positive overflow changes a positive result to a negative value.
        // Negative overflow changes a negative result to a positive value.
        return (a > 0 && b > 0 && static_cast<T>(result) < 0) ||
               (a < 0 && b < 0 && static_cast<T>(result) > 0);
    }
}

template<typename T>
    requires std::is_integral_v<T>
inline bool CheckOverflowSub(T a, T b, T* out) noexcept {
    if constexpr (std::is_unsigned_v<T>) {
        const T result = a - b;
        *out = result;
        return result > a;
    } else {
        using U = std::make_unsigned_t<T>;
        const U result = static_cast<U>(a) - static_cast<U>(b);
        *out = static_cast<T>(result);

        // Subtracting a negative value from a non-negative value can overflow.
        // Subtracting a positive value from a negative value can underflow.
        return (a >= 0 && b < 0 && static_cast<T>(result) < 0) ||
               (a < 0 && b > 0 && static_cast<T>(result) > 0);
    }
}

#endif

/// @brief Multiplies a range of non-negative integers into `*out`.
///
/// Every element in `[first, last)` is processed, and overflow is checked after
/// each multiplication.
///
/// @tparam Iter Input iterator type whose values are convertible to `uint64_t`.
/// @param first Start of the half-open range.
/// @param last End of the half-open range.
/// @param out Destination for the final product.
/// @return True if any multiplication overflowed; false otherwise.
///
/// @pre `out` is not null.
/// @pre Every element is non-negative and convertible to `uint64_t`.
/// @note On overflow, processing continues and `*out` contains the final
///       representable product; callers must ignore that value.
template<typename Iter>
bool SafeMultiplyU64(Iter first, Iter last, uint64_t* out) noexcept {
    AM_DCHECK(out != nullptr);

    uint64_t prod = 1;
    bool overflowed = false;
    while (first != last) {
        using ValueType = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
        if constexpr (std::is_signed_v<ValueType>) {
            AM_DCHECK(*first >= 0);
        }
        overflowed |= CheckOverflowMul(prod, static_cast<uint64_t>(*first), &prod);
        ++first;
    }
    *out = prod;
    return overflowed;
}

/// @brief Multiplies all elements in a container into `*out`.
///
/// @tparam Container Container type providing `begin()` and `end()`.
/// @param c Container whose elements are multiplied.
/// @param out Destination for the final product.
/// @return True if any multiplication overflowed; false otherwise.
/// @pre Every element is non-negative and convertible to `uint64_t`.
template<typename Container>
bool SafeMultiplyU64(const Container& c, uint64_t* out) noexcept {
    return SafeMultiplyU64(c.begin(), c.end(), out);
}

}// namespace aethermind

#endif// AETHERMIND_UTILS_OVERFLOW_CHECK_H
