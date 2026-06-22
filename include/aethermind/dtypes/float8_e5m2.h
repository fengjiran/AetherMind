/// \file
/// 8-bit floating-point type with 5 exponent bits and 2 mantissa bits (E5M2).
///
/// Bit layout: `s eeeee mm` (1 sign | 5 exponent | 2 mantissa, bias 15).
/// E5M2 shares the binary16 exponent range (5 bits, bias 15) but keeps only
/// 2 mantissa bits, so the high 8 bits of a binary16 value and the
/// corresponding E5M2 value are bit-identical. This relationship lets the
/// E5M2 → fp32 conversion reuse the binary16 conversion path.
///
/// Arithmetic is performed by converting to `float`, computing in binary32,
/// and converting back. See https://arxiv.org/pdf/2209.05433.pdf for the
/// reference format definition.

#ifndef AETHERMIND_DTYPES_FLOAT8_E5M2_H
#define AETHERMIND_DTYPES_FLOAT8_E5M2_H

#include "macros.h"

#include <cstdint>
#include <limits>
#include <ostream>

namespace aethermind {
namespace detail {

/// Reinterprets an E5M2 bit pattern as an IEEE 754 binary32 `float`.
///
/// Sign and exponent are preserved exactly. NaN inputs map to a binary32
/// NaN (payload bits placed in the high mantissa positions).
float fp8e5m2_to_fp32_value(uint8_t input);

/// Converts an IEEE 754 binary32 `float` to an E5M2 bit pattern.
///
/// Uses round-to-nearest-even on the truncated mantissa. Values exceeding
/// the E5M2 normal range saturate to ±infinity; values smaller than the
/// minimum normal (2^-14) are encoded as subnormals; NaN inputs are
/// canonicalized to a single quiet-NaN bit pattern.
uint8_t fp8e5m2_from_fp32_value(float f);

}// namespace detail

/// E5M2 8-bit floating-point value.
///
/// Stores the raw 8-bit E5M2 bit pattern in `x`. The format is
/// 1 sign | 5 exponent | 2 mantissa (bias 15), which shares the exponent
/// range of binary16 but has lower precision. Arithmetic and comparisons go
/// through implicit conversion to `float` and follow IEEE 754 binary32
/// rounding and NaN semantics. `is_iec559` is false: E5M2 is not an IEEE 754
/// standard format. Use `from_bits()` to construct from a raw bit pattern
/// without conversion.
struct alignas(1) Float8_e5m2 {
    /// E5M2 bit pattern: 1 sign | 5 exponent | 2 mantissa.
    uint8_t x;

    /// Tag type for constructing a `Float8_e5m2` from raw bits without conversion.
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    Float8_e5m2() : x(0) {}
    /// Constructs from raw E5M2 bits; no floating-point conversion.
    constexpr Float8_e5m2(uint8_t bits, from_bits_t) : x(bits) {}
    Float8_e5m2(float value);// NOLINT
    operator float() const;  // NOLINT

    /// Returns true if the value is positive or negative infinity.
    /// Inf is encoded as exponent=11111 with mantissa=00.
    AM_NODISCARD bool isinf() const;

    /// Returns true if the value is a NaN.
    /// NaN is encoded as exponent=11111 with any non-zero mantissa.
    AM_NODISCARD bool isnan() const;
};

std::ostream& operator<<(std::ostream& os, const Float8_e5m2& value);

/// Arithmetic
Float8_e5m2 operator+(const Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2 operator-(const Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2 operator*(const Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2 operator/(const Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2 operator-(const Float8_e5m2&);
Float8_e5m2& operator+=(Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2& operator-=(Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2& operator*=(Float8_e5m2& lhs, const Float8_e5m2& rhs);
Float8_e5m2& operator/=(Float8_e5m2& lhs, const Float8_e5m2& rhs);

/// Arithmetic with float
float operator+(Float8_e5m2 lhs, float rhs);
float operator-(Float8_e5m2 lhs, float rhs);
float operator*(Float8_e5m2 lhs, float rhs);
float operator/(Float8_e5m2 lhs, float rhs);
float operator+(float lhs, Float8_e5m2 rhs);
float operator-(float lhs, Float8_e5m2 rhs);
float operator*(float lhs, Float8_e5m2 rhs);
float operator/(float lhs, Float8_e5m2 rhs);
float& operator+=(float& lhs, const Float8_e5m2& rhs);
float& operator-=(float& lhs, const Float8_e5m2& rhs);
float& operator*=(float& lhs, const Float8_e5m2& rhs);
float& operator/=(float& lhs, const Float8_e5m2& rhs);

/// Arithmetic with double
double operator+(Float8_e5m2 lhs, double rhs);
double operator-(Float8_e5m2 lhs, double rhs);
double operator*(Float8_e5m2 lhs, double rhs);
double operator/(Float8_e5m2 lhs, double rhs);
double operator+(double lhs, Float8_e5m2 rhs);
double operator-(double lhs, Float8_e5m2 rhs);
double operator*(double lhs, Float8_e5m2 rhs);
double operator/(double lhs, Float8_e5m2 rhs);

/// Arithmetic with int
Float8_e5m2 operator+(Float8_e5m2 lhs, int rhs);
Float8_e5m2 operator-(Float8_e5m2 lhs, int rhs);
Float8_e5m2 operator*(Float8_e5m2 lhs, int rhs);
Float8_e5m2 operator/(Float8_e5m2 lhs, int rhs);
Float8_e5m2 operator+(int lhs, Float8_e5m2 rhs);
Float8_e5m2 operator-(int lhs, Float8_e5m2 rhs);
Float8_e5m2 operator*(int lhs, Float8_e5m2 rhs);
Float8_e5m2 operator/(int lhs, Float8_e5m2 rhs);

/// Arithmetic with int64_t
Float8_e5m2 operator+(Float8_e5m2 lhs, int64_t rhs);
Float8_e5m2 operator-(Float8_e5m2 lhs, int64_t rhs);
Float8_e5m2 operator*(Float8_e5m2 lhs, int64_t rhs);
Float8_e5m2 operator/(Float8_e5m2 lhs, int64_t rhs);
Float8_e5m2 operator+(int64_t lhs, Float8_e5m2 rhs);
Float8_e5m2 operator-(int64_t lhs, Float8_e5m2 rhs);
Float8_e5m2 operator*(int64_t lhs, Float8_e5m2 rhs);
Float8_e5m2 operator/(int64_t lhs, Float8_e5m2 rhs);

}// namespace aethermind

namespace std {

/// std::numeric_limits specialization for E5M2.
///
/// Bit constants follow the E5M2 format (1 sign | 5 exponent | 2 mantissa,
/// bias 15). `is_iec559` is false — E5M2 is not an IEEE 754 standard format.
/// `has_signaling_NaN` is false — E5M2 reserves all `exp=11111, mantissa≠0`
/// patterns for quiet NaN; there is no signaling-NaN encoding.
/// `has_denorm_loss` is true — converting from binary32 routes through a
/// rounding step that can lose denormal precision.
template<>
class numeric_limits<aethermind::Float8_e5m2> {
public:
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_specialized = true;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr auto has_denorm = true;
    static constexpr auto has_denorm_loss = true;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 3;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr aethermind::Float8_e5m2 min() {
        return {0x4, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 max() {
        return {0x7B, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 lowest() {
        return {0xFB, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 epsilon() {
        return {0x34, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 round_error() {
        return {0x38, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 infinity() {
        return {0x7C, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 quiet_NaN() {
        return {0x7F, aethermind::Float8_e5m2::from_bits()};
    }

    static constexpr aethermind::Float8_e5m2 denorm_min() {
        return {0x01, aethermind::Float8_e5m2::from_bits()};
    }
};

}// namespace std

#endif// AETHERMIND_DTYPES_FLOAT8_E5M2_H
