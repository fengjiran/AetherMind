/// \file
/// Software IEEE 754 half-precision (binary16) floating-point type.
///
/// Provides `Half` — a C++ type wrapping a 16-bit binary16 bit pattern — and
/// bit-exact conversion functions between binary16 and binary32. On x86-64
/// with F16C support, conversions use hardware intrinsics via `X86_F16`;
/// otherwise they fall back to portable integer bit manipulation.

#ifndef AETHERMIND_DTYPES_HALF_H
#define AETHERMIND_DTYPES_HALF_H

#include "macros.h"

#include <compare>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#if defined(__aarch64__) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif


#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
        defined(_M_IX86)
#if defined(__F16C__) &&                                   \
        !(defined(__CUDA_ARCH__) || defined(__CUDACC__) || \
          defined(__HIP_DEVICE_COMPILE__))
#define X86_F16 1
#include <immintrin.h>
#endif// defined(__F16C__) && !(defined(__CUDA_ARCH__) || defined(__CUDACC__) \
      // || defined(__HIP_DEVICE_COMPILE__))
#endif// __x86_64__ || _M_X64 || __i386 || _M_IX86
#endif// __GNUC__ || __clang__

namespace aethermind {

/// IEEE 754 half-precision (binary16) floating-point value.
///
/// Wraps a 16-bit binary16 bit pattern. Arithmetic and comparisons go through
/// implicit conversion to `float`, so results follow IEEE 754 binary32 rounding
/// and NaN semantics.
///
/// `is_iec559` is false: `fp16_from_fp32_value` flushes subnormals to zero.
/// Subnormals stored via `from_bits()` are preserved and round-trip correctly
/// through `fp16_to_fp32_value`.
/// Use `from_bits()` to construct from a raw bit pattern without conversion.
struct alignas(2) Half {
    /// Tag type for constructing a `Half` from raw bits without conversion.
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    Half() : x(0) {}

    /// Constructs from raw binary16 bits; no floating-point conversion.
    constexpr Half(uint16_t bits, from_bits_t) : x(bits) {}

    /// Constructs from a binary32 `float`, rounding to binary16.
    ///
    /// Implicit conversion is intentional: it lets `Half` flow through
    /// float-based APIs (literals, arithmetic expressions, printf-style
    /// formatting) with minimal friction. NOLINT suppresses the implicit
    /// conversion warning; callers should still be explicit when ambiguity
    /// matters.
    Half(float value);// NOLINT

    /// Converts to binary32 `float` for arithmetic and comparisons.
    ///
    /// Implicit conversion is intentional: it makes `Half` usable in
    /// float expressions and with the standard library without wrapping
    /// every use site in `static_cast`. NOLINT suppresses the implicit
    /// conversion warning.
    operator float() const;// NOLINT

    /// Returns the raw IEEE 754 binary16 bit pattern.
    AM_NODISCARD constexpr uint16_t bits() const {
        return x;
    }

private:
    /// IEEE 754 binary16 bit pattern: 1 sign | 5 exponent | 10 mantissa.
    uint16_t x;
};

std::ostream& operator<<(std::ostream& os, const Half& value);

/// Arithmetic
Half operator+(const Half& lhs, const Half& rhs);
Half operator-(const Half& lhs, const Half& rhs);
Half operator*(const Half& lhs, const Half& rhs);
Half operator/(const Half& lhs, const Half& rhs);
Half operator-(const Half&);
Half& operator+=(Half& lhs, const Half& rhs);
Half& operator-=(Half& lhs, const Half& rhs);
Half& operator*=(Half& lhs, const Half& rhs);
Half& operator/=(Half& lhs, const Half& rhs);

/// Arithmetic with float
float operator+(Half lhs, float rhs);
float operator-(Half lhs, float rhs);
float operator*(Half lhs, float rhs);
float operator/(Half lhs, float rhs);
float operator+(float lhs, Half rhs);
float operator-(float lhs, Half rhs);
float operator*(float lhs, Half rhs);
float operator/(float lhs, Half rhs);
float& operator+=(float& lhs, const Half& rhs);
float& operator-=(float& lhs, const Half& rhs);
float& operator*=(float& lhs, const Half& rhs);
float& operator/=(float& lhs, const Half& rhs);

/// Arithmetic with double
double operator+(Half lhs, double rhs);
double operator-(Half lhs, double rhs);
double operator*(Half lhs, double rhs);
double operator/(Half lhs, double rhs);
double operator+(double lhs, Half rhs);
double operator-(double lhs, Half rhs);
double operator*(double lhs, Half rhs);
double operator/(double lhs, Half rhs);

/// Arithmetic with int (result is float, matching Half-with-float promotion).
float operator+(Half lhs, int rhs);
float operator-(Half lhs, int rhs);
float operator*(Half lhs, int rhs);
float operator/(Half lhs, int rhs);
float operator+(int lhs, Half rhs);
float operator-(int lhs, Half rhs);
float operator*(int lhs, Half rhs);
float operator/(int lhs, Half rhs);

/// Arithmetic with int64_t (result is float, matching Half-with-float promotion).
float operator+(Half lhs, int64_t rhs);
float operator-(Half lhs, int64_t rhs);
float operator*(Half lhs, int64_t rhs);
float operator/(Half lhs, int64_t rhs);
float operator+(int64_t lhs, Half rhs);
float operator-(int64_t lhs, Half rhs);
float operator*(int64_t lhs, Half rhs);
float operator/(int64_t lhs, Half rhs);

/// Comparison operators based on IEEE 754 binary32 value ordering.
///
/// These are constrained templates so that only operands whose actual type
/// is exactly `Half` participate in overload resolution. This avoids ambiguity
/// with built-in comparisons such as `long < Half`, where `Half` could otherwise
/// be reached through both its implicit `float` conversion and a user-defined
/// conversion from the other operand.
///
/// NaN is unordered relative to any value, including itself.
/// +0.0 and -0.0 compare equivalent.
template<typename T, typename U>
    requires(std::same_as<std::remove_cvref_t<T>, Half> && std::same_as<std::remove_cvref_t<U>, Half>)
AM_NODISCARD std::partial_ordering operator<=>(T&& lhs, U&& rhs) {
    return static_cast<float>(lhs) <=> static_cast<float>(rhs);
}

template<typename T, typename U>
    requires(std::same_as<std::remove_cvref_t<T>, Half> && std::same_as<std::remove_cvref_t<U>, Half>)
AM_NODISCARD bool operator==(T&& lhs, U&& rhs) {
    return static_cast<float>(lhs) == static_cast<float>(rhs);
}

}// namespace aethermind

/// std::numeric_limits specialization for IEEE 754 binary16.
///
/// Bit constants follow the binary16 format (1 sign | 5 exponent | 10 mantissa,
/// bias 15). `is_iec559` is false: `fp16_from_fp32_value` flushes subnormals to
/// zero, so the type does not provide gradual underflow. Subnormals stored via
/// `from_bits()` are preserved and round-trip correctly through
/// `fp16_to_fp32_value`.
template<>
struct std::numeric_limits<aethermind::Half> {
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = denorm_present;
    static constexpr auto has_denorm_loss = true;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr aethermind::Half min() {
        return {0x0400, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half lowest() {
        return {0xFBFF, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half max() {
        return {0x7BFF, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half epsilon() {
        return {0x1400, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half round_error() {
        return {0x3800, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half infinity() {
        return {0x7C00, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half quiet_NaN() {
        return {0x7E00, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half signaling_NaN() {
        return {0x7D00, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half denorm_min() {
        return {0x0001, aethermind::Half::from_bits()};
    }
};// namespace std

#endif// AETHERMIND_DTYPES_HALF_H
