/// \file
/// Software IEEE 754 half-precision (binary16) floating-point type.
///
/// Provides `Half` — a C++ type wrapping a 16-bit binary16 bit pattern — and
/// bit-exact conversion functions between binary16 and binary32. On x86-64
/// with F16C support, conversions use hardware intrinsics via `X86_F16`;
/// otherwise they fall back to portable integer bit manipulation.

#ifndef AETHERMIND_HALF_H
#define AETHERMIND_HALF_H

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <ostream>

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
namespace details {

/// Converts an IEEE half-precision value (bit representation) to an IEEE
/// single-precision value (bit representation).
///
/// Implemented with integer bit manipulation only; no floating-point operations.
uint32_t fp16_to_fp32_bits(uint16_t h);

/// Converts an IEEE half-precision value (bit representation) to an IEEE
/// single-precision `float`.
///
/// Uses IEEE-like floating-point operations and int/float bitcasts; makes no
/// assumption about the current rounding mode and performs no operations on
/// denormals.
float fp16_to_fp32_value(uint16_t h);

/// Converts an IEEE single-precision `float` to an IEEE half-precision value
/// (bit representation).
///
/// Implemented with integer bit manipulation only; no floating-point operations.
/// Rounding mode is round-to-nearest-even.
uint16_t fp16_from_fp32_value(float f);

}// namespace details

/// IEEE 754 half-precision (binary16) floating-point value.
///
/// Stores the raw 16-bit binary16 bit pattern in `x`. Arithmetic and
/// comparisons go through implicit conversion to `float`, so results follow
/// IEEE 754 binary32 rounding and NaN semantics. `is_iec559` is true.
/// Use `from_bits()` to construct from a raw bit pattern without conversion.
struct alignas(2) Half {
    /// IEEE 754 binary16 bit pattern: 1 sign | 5 exponent | 10 mantissa.
    uint16_t x;

    /// Tag type for constructing a `Half` from raw bits without conversion.
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    Half() : x(0) {}
    /// Constructs from raw binary16 bits; no floating-point conversion.
    constexpr Half(uint16_t bits, from_bits_t) : x(bits) {}
    Half(float value);// NOLINT

    operator float() const;// NOLINT
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

/// Arithmetic with int
Half operator+(Half lhs, int rhs);
Half operator-(Half lhs, int rhs);
Half operator*(Half lhs, int rhs);
Half operator/(Half lhs, int rhs);
Half operator+(int lhs, Half rhs);
Half operator-(int lhs, Half rhs);
Half operator*(int lhs, Half rhs);
Half operator/(int lhs, Half rhs);

/// Arithmetic with int64_t
Half operator+(Half lhs, int64_t rhs);
Half operator-(Half lhs, int64_t rhs);
Half operator*(Half lhs, int64_t rhs);
Half operator/(Half lhs, int64_t rhs);
Half operator+(int64_t lhs, Half rhs);
Half operator-(int64_t lhs, Half rhs);
Half operator*(int64_t lhs, Half rhs);
Half operator/(int64_t lhs, Half rhs);

}// namespace aethermind

/// std::numeric_limits specialization for IEEE 754 binary16.
///
/// Bit constants follow the binary16 format (1 sign | 5 exponent | 10 mantissa,
/// bias 15). `is_iec559` is true: Half follows IEEE 754 NaN and denormal
/// semantics.
template<>
struct std::numeric_limits<aethermind::Half> {
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = true;
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

#endif// AETHERMIND_HALF_H
