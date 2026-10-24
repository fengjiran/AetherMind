#ifndef AETHERMIND_UTILS_OVERFLOW_CHECK_H
#define AETHERMIND_UTILS_OVERFLOW_CHECK_H

/// @file
/// @brief Checked integer arithmetic, conversions, and range-product utilities.
///
/// Provides checked arithmetic operations that report overflow without relying
/// on undefined signed-overflow behavior. Uses compiler builtins when available
/// (GCC/Clang), otherwise falls back to portable implementations.

#include "aethermind/base/macros.h"
#include "utils/logging.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace aethermind {

namespace detail {

template<typename T>
concept CheckNarrowInteger =
        std::is_same_v<T, signed char> ||
        std::is_same_v<T, unsigned char> ||
        std::is_same_v<T, short> ||
        std::is_same_v<T, unsigned short> ||
        std::is_same_v<T, int> ||
        std::is_same_v<T, unsigned int> ||
        std::is_same_v<T, long> ||
        std::is_same_v<T, unsigned long> ||
        std::is_same_v<T, long long> ||
        std::is_same_v<T, unsigned long long>;

template<typename T>
concept OverflowInteger = std::is_integral_v<T> &&
                          !std::is_same_v<T, bool> &&
                          std::is_same_v<T, std::remove_cv_t<T>>;

// At least unsigned int is needed: unsigned short operands may otherwise be
// promoted to signed int before multiplication, introducing signed overflow.
template<OverflowInteger T>
using OverflowUnsignedArithmetic = decltype(std::make_unsigned_t<T>{} + 0U);

template<OverflowInteger T>
constexpr bool CheckOverflowMulPortable(T a, T b, T* out) noexcept {
    using Limits = std::numeric_limits<T>;
    bool overflowed = false;
    if constexpr (std::is_signed_v<T>) {
        if (a > 0) {
            overflowed = b > 0 ? a > Limits::max() / b : b < Limits::min() / a;
        } else if (a < 0) {
            overflowed = b > 0 ? a < Limits::min() / b : b < 0 && a < Limits::max() / b;
        }
    } else {
        overflowed = b != 0 && a > Limits::max() / b;
    }

    using U = OverflowUnsignedArithmetic<T>;
    *out = static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
    return overflowed;
}

template<OverflowInteger T>
constexpr bool CheckOverflowAddPortable(T a, T b, T* out) noexcept {
    using Limits = std::numeric_limits<T>;
    bool overflowed = false;
    if constexpr (std::is_signed_v<T>) {
        overflowed = (b > 0 && a > Limits::max() - b) ||
                     (b < 0 && a < Limits::min() - b);
    } else {
        overflowed = a > Limits::max() - b;
    }

    using U = OverflowUnsignedArithmetic<T>;
    *out = static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
    return overflowed;
}

template<OverflowInteger T>
constexpr bool CheckOverflowSubPortable(T a, T b, T* out) noexcept {
    using Limits = std::numeric_limits<T>;
    bool overflowed = false;
    if constexpr (std::is_signed_v<T>) {
        overflowed = (b > 0 && a < Limits::min() + b) ||
                     (b < 0 && a > Limits::max() + b);
    } else {
        overflowed = a < b;
    }

    using U = OverflowUnsignedArithmetic<T>;
    *out = static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
    return overflowed;
}

} // namespace detail

/// @brief Checks representability and converts between standard integer types.
/// @tparam To Unqualified destination integer type.
/// @tparam From Source integer type.
/// @param value Value to convert.
/// @param out Receives the converted value on success.
/// @return True when value is not representable in To; false on success.
/// @pre out is not null.
/// @note Failure leaves *out unchanged. Widening conversions are also supported.
template<detail::CheckNarrowInteger To, detail::CheckNarrowInteger From>
AM_NODISCARD constexpr bool CheckNarrow(From value, To* out) noexcept {
    if (!std::in_range<To>(value)) {
        return true;
    }

    *out = static_cast<To>(value);
    return false;
}

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
template<detail::OverflowInteger T>
bool CheckOverflowMul(T a, T b, T* out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    return detail::CheckOverflowMulPortable(a, b, out);
#endif
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
template<detail::OverflowInteger T>
bool CheckOverflowAdd(T a, T b, T* out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    return detail::CheckOverflowAddPortable(a, b, out);
#endif
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
template<detail::OverflowInteger T>
bool CheckOverflowSub(T a, T b, T* out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_sub_overflow(a, b, out);
#else
    return detail::CheckOverflowSubPortable(a, b, out);
#endif
}

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

} // namespace aethermind

#endif // AETHERMIND_UTILS_OVERFLOW_CHECK_H
