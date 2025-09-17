//
// Created by 赵丹 on 2025/9/12.
//
#include "utils/float8_e4m3fn.h"

namespace aethermind {
namespace details {

static uint32_t fp8e4m3fn_to_fp32_bits(uint8_t input) {
    const uint32_t w = static_cast<uint32_t>(input) << 24;

    const uint32_t sign = w & UINT32_C(0x80000000);

    const uint32_t exponent = w & UINT32_C(0x78000000);
    const uint32_t mantissa = w & UINT32_C(0x07000000);

    // zero
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }

    // nan
    if (exponent == 0x78000000 && mantissa == 0x07000000) {
        return sign | 0x7F800000 | mantissa >> 4;
    }

    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
    return sign | (nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23);
}

float fp8e4m3fn_to_fp32_value(uint8_t input) {
    return fp32_from_bits(fp8e4m3fn_to_fp32_bits(input));
}

uint8_t fp8e4m3fn_from_fp32_value(float f) {
    uint32_t x = fp32_to_bits(f);
    const uint32_t sign = x & UINT32_C(0x80000000);
    const uint32_t exponent = x & UINT32_C(0x7F800000);
    const uint32_t mantissa = x & UINT32_C(0x007FFFFF);

    // zero
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint8_t>(sign >> 24);
    }

    uint32_t nonsign = exponent | mantissa;

    // convert inf, nan and greater than nan(480.0f) in fp32 to nan
    if (exponent == UINT32_C(0x7F800000) || nonsign >= UINT32_C(0x43F00000)) {
        return static_cast<uint8_t>(sign >> 24 | 0x7F);
    }

    if (nonsign < UINT32_C(121) << 23) {
        // The input number is smaller than 2^(-6), which is the smallest
        // fp8e4m3fn normal number, convert to denormal number
        uint32_t denorm_mask = UINT32_C(141) << 23;
        nonsign = fp32_to_bits(fp32_from_bits(nonsign) + fp32_from_bits(denorm_mask));
        return static_cast<uint8_t>(nonsign - denorm_mask) | static_cast<uint8_t>(sign >> 24);
    }

    // normalize the exponent from fp32 bias(127) to fp8 bias(7)
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    // convert to fp8 format
    uint32_t res = sign >> 24;

    // add fp8 bias (7) to exponent
    res |= static_cast<uint32_t>(exp32 + 7) << 3;

    // add mantissa (round to the nearest even)
    res |= mantissa >> 20;

    // handle rounding
    const uint32_t rounding_bit = mantissa & UINT32_C(0x00080000);
    const uint32_t sticky_bits = mantissa & UINT32_C(0x0007FFFF);

    if (rounding_bit && (sticky_bits || (res & 1))) {
        res += 1;
    }

    return static_cast<uint8_t>(res);
}
}// namespace details

Float8_e4m3fn::Float8_e4m3fn(float value) : x(details::fp8e4m3fn_from_fp32_value(value)) {}

Float8_e4m3fn::operator float() const {
    return details::fp8e4m3fn_to_fp32_value(x);
}

bool Float8_e4m3fn::isnan() const {
    return (x & 0x7F) == 0x7F;
}

std::ostream& operator<<(std::ostream& os, const Float8_e4m3fn& value) {
    os << static_cast<float>(value);
    return os;
}

Float8_e4m3fn operator+(const Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

Float8_e4m3fn operator-(const Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

Float8_e4m3fn operator*(const Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

Float8_e4m3fn operator/(const Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

Float8_e4m3fn operator-(const Float8_e4m3fn& a) {
    return -static_cast<float>(a);
}

Float8_e4m3fn& operator+=(Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

Float8_e4m3fn& operator-=(Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

Float8_e4m3fn& operator*=(Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

Float8_e4m3fn& operator/=(Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs) {
    lhs = lhs / rhs;
    return lhs;
}

float operator+(Float8_e4m3fn lhs, float rhs) {
    return static_cast<float>(lhs) + rhs;
}

float operator-(Float8_e4m3fn lhs, float rhs) {
    return static_cast<float>(lhs) - rhs;
}

float operator*(Float8_e4m3fn lhs, float rhs) {
    return static_cast<float>(lhs) * rhs;
}

float operator/(Float8_e4m3fn lhs, float rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / rhs;
}

float operator+(float lhs, Float8_e4m3fn rhs) {
    return lhs + static_cast<float>(rhs);
}

float operator-(float lhs, Float8_e4m3fn rhs) {
    return lhs - static_cast<float>(rhs);
}

float operator*(float lhs, Float8_e4m3fn rhs) {
    return lhs * static_cast<float>(rhs);
}

float operator/(float lhs, Float8_e4m3fn rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<float>(rhs);
}

float& operator+=(float& lhs, const Float8_e4m3fn& rhs) {
    return lhs += static_cast<float>(rhs);
}

float& operator-=(float& lhs, const Float8_e4m3fn& rhs) {
    return lhs -= static_cast<float>(rhs);
}

float& operator*=(float& lhs, const Float8_e4m3fn& rhs) {
    return lhs *= static_cast<float>(rhs);
}

float& operator/=(float& lhs, const Float8_e4m3fn& rhs) {
    return lhs /= static_cast<float>(rhs);
}

double operator+(Float8_e4m3fn lhs, double rhs) {
    return static_cast<double>(lhs) + rhs;
}

double operator-(Float8_e4m3fn lhs, double rhs) {
    return static_cast<double>(lhs) - rhs;
}

double operator*(Float8_e4m3fn lhs, double rhs) {
    return static_cast<double>(lhs) * rhs;
}

double operator/(Float8_e4m3fn lhs, double rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<double>(lhs) / rhs;
}

double operator+(double lhs, Float8_e4m3fn rhs) {
    return lhs + static_cast<double>(rhs);
}

double operator-(double lhs, Float8_e4m3fn rhs) {
    return lhs - static_cast<double>(rhs);
}

double operator*(double lhs, Float8_e4m3fn rhs) {
    return lhs * static_cast<double>(rhs);
}

double operator/(double lhs, Float8_e4m3fn rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<double>(rhs);
}

Float8_e4m3fn operator+(Float8_e4m3fn lhs, int rhs) {
    return lhs + static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator-(Float8_e4m3fn lhs, int rhs) {
    return lhs - static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator*(Float8_e4m3fn lhs, int rhs) {
    return lhs * static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator/(Float8_e4m3fn lhs, int rhs) {
    return lhs / static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator+(int lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) + rhs;
}

Float8_e4m3fn operator-(int lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) / rhs;
}

Float8_e4m3fn operator*(int lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) * rhs;
}

Float8_e4m3fn operator/(int lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) / rhs;
}

Float8_e4m3fn operator+(Float8_e4m3fn lhs, int64_t rhs) {
    return lhs + static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator-(Float8_e4m3fn lhs, int64_t rhs) {
    return lhs - static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator*(Float8_e4m3fn lhs, int64_t rhs) {
    return lhs * static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator/(Float8_e4m3fn lhs, int64_t rhs) {
    return lhs / static_cast<Float8_e4m3fn>(rhs);
}

Float8_e4m3fn operator+(int64_t lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) + rhs;
}

Float8_e4m3fn operator-(int64_t lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) - rhs;
}

Float8_e4m3fn operator*(int64_t lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) * rhs;
}

Float8_e4m3fn operator/(int64_t lhs, Float8_e4m3fn rhs) {
    return static_cast<Float8_e4m3fn>(lhs) / rhs;
}


}// namespace aethermind