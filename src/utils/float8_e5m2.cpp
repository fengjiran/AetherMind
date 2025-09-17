//
// Created by 赵丹 on 2025/9/17.
//

#include "utils/float8_e5m2.h"

namespace aethermind {
namespace details {

float fp8e5m2_to_fp32_value(uint8_t input) {
    uint16_t h = input;
    h <<= 8;
    return half_to_fp32_value_ieee(h);
}

uint8_t fp8e5m2_from_fp32_value(float f) {
    uint32_t x = fp32_to_bits(f);
    const uint32_t sign = x & UINT32_C(0x80000000);
    const uint32_t exponent = x & UINT32_C(0x7F800000);
    const uint32_t mantissa = x & UINT32_C(0x007FFFFF);

    // zero
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint8_t>(sign >> 24);
    }

    // inf and nan case
    if (exponent == UINT32_C(0x7F800000)) {
        if (mantissa == 0) {
            return static_cast<uint8_t>(sign >> 24 | 0x7C);
        }
        return static_cast<uint8_t>(sign >> 24 | 0x7E);
    }

    uint32_t nonsign = exponent | mantissa;
    if (nonsign >= UINT32_C(0x47800000)) {
        return nonsign > UINT32_C(0x7F800000) ? static_cast<uint8_t>(sign >> 24 | 0x7F) : static_cast<uint8_t>(sign >> 24 | 0x7C);
    }

    if (nonsign < UINT32_C(113) << 23) {
        // The input number is smaller than 2^(-14), which is the smallest
        // fp8e5m2 normal number, convert to denormal number
        uint32_t denorm_mask = UINT32_C(134) << 23;
        nonsign = fp32_to_bits(fp32_from_bits(nonsign) + fp32_from_bits(denorm_mask));
        return static_cast<uint8_t>(nonsign - denorm_mask) | static_cast<uint8_t>(sign >> 24);
    }

    // normalize the exponent from fp32 bias(127) to fp8 bias(15)
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    // convert to fp8 format
    uint32_t res = sign >> 24;

    // add fp8 bias (15) to exponent
    res |= static_cast<uint32_t>(exp32 + 15) << 2;

    // add mantissa (round to the nearest even)
    res |= mantissa >> 21;

    // handle rounding
    const uint32_t rounding_bit = mantissa & UINT32_C(0x00100000);
    const uint32_t sticky_bits = mantissa & UINT32_C(0x000FFFFF);

    if (rounding_bit && (sticky_bits || (res & 1))) {
        res += 1;
    }

    return static_cast<uint8_t>(res);
}

}// namespace details

Float8_e5m2::Float8_e5m2(float value) : x(details::fp8e5m2_from_fp32_value(value)) {}

Float8_e5m2::operator float() const {
    return details::fp8e5m2_to_fp32_value(x);
}

bool Float8_e5m2::isinf() const {
    return (x & 0x7F) == 0x7C;
}

bool Float8_e5m2::isnan() const {
    return (x & 0x7F) > 0x7C;
}

std::ostream& operator<<(std::ostream& os, const Float8_e5m2& value) {
    os << static_cast<float>(value);
    return os;
}

Float8_e5m2 operator+(const Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

Float8_e5m2 operator-(const Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

Float8_e5m2 operator*(const Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

Float8_e5m2 operator/(const Float8_e5m2& lhs, const Float8_e5m2& rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

Float8_e5m2 operator-(const Float8_e5m2& a) {
    return -static_cast<float>(a);
}

Float8_e5m2& operator+=(Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

Float8_e5m2& operator-=(Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

Float8_e5m2& operator*=(Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

Float8_e5m2& operator/=(Float8_e5m2& lhs, const Float8_e5m2& rhs) {
    lhs = lhs / rhs;
    return lhs;
}

float operator+(Float8_e5m2 lhs, float rhs) {
    return static_cast<float>(lhs) + rhs;
}

float operator-(Float8_e5m2 lhs, float rhs) {
    return static_cast<float>(lhs) + rhs;
}

float operator*(Float8_e5m2 lhs, float rhs) {
    return static_cast<float>(lhs) * rhs;
}

float operator/(Float8_e5m2 lhs, float rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / rhs;
}

float operator+(float lhs, Float8_e5m2 rhs) {
    return lhs + static_cast<float>(rhs);
}

float operator-(float lhs, Float8_e5m2 rhs) {
    return lhs - static_cast<float>(rhs);
}

float operator*(float lhs, Float8_e5m2 rhs) {
    return lhs * static_cast<float>(rhs);
}

float operator/(float lhs, Float8_e5m2 rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<float>(rhs);
}

float& operator+=(float& lhs, const Float8_e5m2& rhs) {
    return lhs += static_cast<float>(rhs);
}

float& operator-=(float& lhs, const Float8_e5m2& rhs) {
    return lhs -= static_cast<float>(rhs);
}

float& operator*=(float& lhs, const Float8_e5m2& rhs) {
    return lhs *= static_cast<float>(rhs);
}

float& operator/=(float& lhs, const Float8_e5m2& rhs) {
    return lhs /= static_cast<float>(rhs);
}

double operator+(Float8_e5m2 lhs, double rhs) {
    return static_cast<double>(lhs) + rhs;
}

double operator-(Float8_e5m2 lhs, double rhs) {
    return static_cast<double>(lhs) - rhs;
}

double operator*(Float8_e5m2 lhs, double rhs) {
    return static_cast<double>(lhs) * rhs;
}

double operator/(Float8_e5m2 lhs, double rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<double>(lhs) / rhs;
}

double operator+(double lhs, Float8_e5m2 rhs) {
    return lhs + static_cast<double>(rhs);
}

double operator-(double lhs, Float8_e5m2 rhs) {
    return lhs - static_cast<double>(rhs);
}

double operator*(double lhs, Float8_e5m2 rhs) {
    return lhs * static_cast<double>(rhs);
}

double operator/(double lhs, Float8_e5m2 rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<double>(rhs);
}

Float8_e5m2 operator+(Float8_e5m2 lhs, int rhs) {
    return lhs + static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator-(Float8_e5m2 lhs, int rhs) {
    return lhs - static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator*(Float8_e5m2 lhs, int rhs) {
    return lhs * static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator/(Float8_e5m2 lhs, int rhs) {
    return lhs / static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator+(int lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) + rhs;
}

Float8_e5m2 operator-(int lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) - rhs;
}

Float8_e5m2 operator*(int lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) * rhs;
}

Float8_e5m2 operator/(int lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) / rhs;
}

Float8_e5m2 operator+(Float8_e5m2 lhs, int64_t rhs) {
    return lhs + static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator-(Float8_e5m2 lhs, int64_t rhs) {
    return lhs - static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator*(Float8_e5m2 lhs, int64_t rhs) {
    return lhs * static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator/(Float8_e5m2 lhs, int64_t rhs) {
    return lhs / static_cast<Float8_e5m2>(rhs);
}

Float8_e5m2 operator+(int64_t lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) + rhs;
}

Float8_e5m2 operator-(int64_t lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) - rhs;
}

Float8_e5m2 operator*(int64_t lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) * rhs;
}

Float8_e5m2 operator/(int64_t lhs, Float8_e5m2 rhs) {
    return static_cast<Float8_e5m2>(lhs) / rhs;
}
}// namespace aethermind