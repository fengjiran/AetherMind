//
// Created by 赵丹 on 2025/9/17.
//

#ifndef AETHERMIND_FLOAT8_E5M2_H
#define AETHERMIND_FLOAT8_E5M2_H

/// Defines the Float8_e5m2 type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeeee mm
/// 1 sign bit
/// 5 exponent bits
/// 2 mantissa bits
/// bias = 15
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf

#include "utils/half.h"

namespace aethermind {
namespace details {

/*
 * Convert a 8-bit floating-point number in fp8 E5M2 format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
float fp8e5m2_to_fp32_value(uint8_t input);

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
uint8_t fp8e5m2_from_fp32_value(float f);

}// namespace details

struct alignas(1) Float8_e5m2 {
    uint8_t x;

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    Float8_e5m2() : x(0) {}
    constexpr Float8_e5m2(uint8_t bits, from_bits_t) : x(bits) {}
    Float8_e5m2(float value);// NOLINT
    operator float() const;  // NOLINT
    NODISCARD bool isinf() const;
    NODISCARD bool isnan() const;
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

#endif//AETHERMIND_FLOAT8_E5M2_H
