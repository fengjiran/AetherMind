//
// Created by richard on 9/23/25.
//

#ifndef AETHERMIND_BFLOAT16_H
#define AETHERMIND_BFLOAT16_H

#include "macros.h"
#include "utils/floating_point_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <ostream>

namespace aethermind {
namespace details {

float bf16_to_fp32_value(uint16_t input);

uint16_t bf16_from_fp32_value(float);

}// namespace details

struct alignas(2) BFloat16 {
    uint16_t x;

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    BFloat16() : x(0) {}
    constexpr BFloat16(uint16_t bits, from_bits_t) : x(bits) {}
    BFloat16(float);       //NOLINT
    operator float() const;//NOLINT
};

std::ostream& operator<<(std::ostream& os, const BFloat16& value);

/// Arithmetic
BFloat16 operator+(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator-(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator*(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator/(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator-(const BFloat16&);
BFloat16& operator+=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator-=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator*=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator/=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator|(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator^(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator&(BFloat16& lhs, const BFloat16& rhs);

/// Arithmetic with float
float operator+(BFloat16 lhs, float rhs);
float operator-(BFloat16 lhs, float rhs);
float operator*(BFloat16 lhs, float rhs);
float operator/(BFloat16 lhs, float rhs);
float operator+(float lhs, BFloat16 rhs);
float operator-(float lhs, BFloat16 rhs);
float operator*(float lhs, BFloat16 rhs);
float operator/(float lhs, BFloat16 rhs);
float& operator+=(float& lhs, const BFloat16& rhs);
float& operator-=(float& lhs, const BFloat16& rhs);
float& operator*=(float& lhs, const BFloat16& rhs);
float& operator/=(float& lhs, const BFloat16& rhs);

/// Arithmetic with double
double operator+(BFloat16 lhs, double rhs);
double operator-(BFloat16 lhs, double rhs);
double operator*(BFloat16 lhs, double rhs);
double operator/(BFloat16 lhs, double rhs);
double operator+(double lhs, BFloat16 rhs);
double operator-(double lhs, BFloat16 rhs);
double operator*(double lhs, BFloat16 rhs);
double operator/(double lhs, BFloat16 rhs);

/// Arithmetic with int
BFloat16 operator+(BFloat16 lhs, int rhs);
BFloat16 operator-(BFloat16 lhs, int rhs);
BFloat16 operator*(BFloat16 lhs, int rhs);
BFloat16 operator/(BFloat16 lhs, int rhs);
BFloat16 operator+(int lhs, BFloat16 rhs);
BFloat16 operator-(int lhs, BFloat16 rhs);
BFloat16 operator*(int lhs, BFloat16 rhs);
BFloat16 operator/(int lhs, BFloat16 rhs);

/// Arithmetic with int64_t
BFloat16 operator+(BFloat16 lhs, int64_t rhs);
BFloat16 operator-(BFloat16 lhs, int64_t rhs);
BFloat16 operator*(BFloat16 lhs, int64_t rhs);
BFloat16 operator/(BFloat16 lhs, int64_t rhs);
BFloat16 operator+(int64_t lhs, BFloat16 rhs);
BFloat16 operator-(int64_t lhs, BFloat16 rhs);
BFloat16 operator*(int64_t lhs, BFloat16 rhs);
BFloat16 operator/(int64_t lhs, BFloat16 rhs);

bool operator>(const BFloat16& lhs, const BFloat16& rhs);
bool operator<(const BFloat16& lhs, const BFloat16& rhs);

}// namespace aethermind

namespace std {

template<>
class numeric_limits<aethermind::BFloat16> {
public:
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr aethermind::BFloat16 min() {
        return {0x0080, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 lowest() {
        return {0xFF7F, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 max() {
        return {0x7F7F, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 epsilon() {
        return {0x3C00, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 round_error() {
        return {0x3F00, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 infinity() {
        return {0x7F80, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 quiet_NaN() {
        return {0x7FC0, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 signaling_NaN() {
        return {0x7F80, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 denorm_min() {
        return {0x0001, aethermind::BFloat16::from_bits()};
    }
};

}// namespace std

#endif//AETHERMIND_BFLOAT16_H
