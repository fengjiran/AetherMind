/// \file
/// Implementation of `BFloat16` and bfloat16 ↔ binary32 conversion.

#include "utils/bfloat16.h"

#include "macros.h"

namespace aethermind {
namespace details {

float bf16_to_fp32_value(uint16_t input) {
    // bfloat16 occupies the high 16 bits of a binary32: shifting left by 16
    // and reinterpreting yields the corresponding float without any rounding.
    uint32_t tmp = input;
    tmp <<= 16;
    return fp32_from_bits(tmp);
}

uint16_t bf16_from_fp32_value(float value) {
    // Canonicalize all NaNs to a single quiet-NaN bit pattern. 0x7FC0 has
    // sign=0, exponent=0xFF, mantissa MSB=1 (quiet), other mantissa bits=0.
    if (std::isnan(value)) {
        return UINT16_C(0x7FC0);
    }

    // Round-to-nearest-even via rounding bias: bias the lower 16 bits before
    // truncation so that the high 16 bits round correctly. The bias depends
    // on the LSB of the surviving high half — when that LSB is 1, ties round
    // up; when 0, ties round down. This makes ties round to the even value.
    uint32_t x = fp32_to_bits(value);
    uint32_t rounding_bias = (x >> 16 & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

}// namespace details

BFloat16::BFloat16(float value) : x(details::bf16_from_fp32_value(value)) {}

BFloat16::operator float() const {
    return details::bf16_to_fp32_value(x);
}

std::ostream& operator<<(std::ostream& os, const BFloat16& value) {
    os << static_cast<float>(value);
    return os;
}

BFloat16 operator+(const BFloat16& lhs, const BFloat16& rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

BFloat16 operator-(const BFloat16& lhs, const BFloat16& rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

BFloat16 operator*(const BFloat16& lhs, const BFloat16& rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

BFloat16 operator/(const BFloat16& lhs, const BFloat16& rhs)
        // IEEE 754 defines float division by zero as ±inf; suppress UBSan
        // which treats it as undefined behavior.
        __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

BFloat16 operator-(const BFloat16& a) {
    return -static_cast<float>(a);
}

BFloat16& operator+=(BFloat16& lhs, const BFloat16& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

BFloat16& operator-=(BFloat16& lhs, const BFloat16& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

BFloat16& operator*=(BFloat16& lhs, const BFloat16& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

BFloat16& operator/=(BFloat16& lhs, const BFloat16& rhs) {
    lhs = lhs / rhs;
    return lhs;
}

BFloat16& operator|(BFloat16& lhs, const BFloat16& rhs) {
    lhs.x |= rhs.x;
    return lhs;
}

BFloat16& operator&(BFloat16& lhs, const BFloat16& rhs) {
    lhs.x &= rhs.x;
    return lhs;
}

BFloat16& operator^(BFloat16& lhs, const BFloat16& rhs) {
    lhs.x ^= rhs.x;
    return lhs;
}

float operator+(BFloat16 lhs, float rhs) {
    return static_cast<float>(lhs) + rhs;
}

float operator-(BFloat16 lhs, float rhs) {
    return static_cast<float>(lhs) - rhs;
}

float operator*(BFloat16 lhs, float rhs) {
    return static_cast<float>(lhs) * rhs;
}

float operator/(BFloat16 lhs, float rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / rhs;
}

float operator+(float lhs, BFloat16 rhs) {
    return lhs + static_cast<float>(rhs);
}

float operator-(float lhs, BFloat16 rhs) {
    return lhs - static_cast<float>(rhs);
}

float operator*(float lhs, BFloat16 rhs) {
    return lhs * static_cast<float>(rhs);
}

float operator/(float lhs, BFloat16 rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<float>(rhs);
}

float& operator+=(float& lhs, const BFloat16& rhs) {
    return lhs += static_cast<float>(rhs);
}

float& operator-=(float& lhs, const BFloat16& rhs) {
    return lhs -= static_cast<float>(rhs);
}

float& operator*=(float& lhs, const BFloat16& rhs) {
    return lhs *= static_cast<float>(rhs);
}

float& operator/=(float& lhs, const BFloat16& rhs) {
    return lhs /= static_cast<float>(rhs);
}

double operator+(BFloat16 lhs, double rhs) {
    return static_cast<double>(lhs) + rhs;
}

double operator-(BFloat16 lhs, double rhs) {
    return static_cast<double>(lhs) - rhs;
}

double operator*(BFloat16 lhs, double rhs) {
    return static_cast<double>(lhs) * rhs;
}

double operator/(BFloat16 lhs, double rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<double>(lhs) / rhs;
}

double operator+(double lhs, BFloat16 rhs) {
    return lhs + static_cast<double>(rhs);
}

double operator-(double lhs, BFloat16 rhs) {
    return lhs - static_cast<double>(rhs);
}

double operator*(double lhs, BFloat16 rhs) {
    return lhs * static_cast<double>(rhs);
}

double operator/(double lhs, BFloat16 rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<double>(rhs);
}

BFloat16 operator+(BFloat16 lhs, int rhs) {
    return lhs + static_cast<BFloat16>(rhs);
}

BFloat16 operator-(BFloat16 lhs, int rhs) {
    return lhs - static_cast<BFloat16>(rhs);
}

BFloat16 operator*(BFloat16 lhs, int rhs) {
    return lhs * static_cast<BFloat16>(rhs);
}

BFloat16 operator/(BFloat16 lhs, int rhs) {
    return lhs / static_cast<BFloat16>(rhs);
}

BFloat16 operator+(int lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) + rhs;
}

BFloat16 operator-(int lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) - rhs;
}

BFloat16 operator*(int lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) * rhs;
}

BFloat16 operator/(int lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) / rhs;
}

BFloat16 operator+(BFloat16 lhs, int64_t rhs) {
    return lhs + static_cast<BFloat16>(rhs);
}

BFloat16 operator-(BFloat16 lhs, int64_t rhs) {
    return lhs - static_cast<BFloat16>(rhs);
}

BFloat16 operator*(BFloat16 lhs, int64_t rhs) {
    return lhs * static_cast<BFloat16>(rhs);
}

BFloat16 operator/(BFloat16 lhs, int64_t rhs) {
    return lhs / static_cast<BFloat16>(rhs);
}

BFloat16 operator+(int64_t lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) + rhs;
}

BFloat16 operator-(int64_t lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) - rhs;
}

BFloat16 operator*(int64_t lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) * rhs;
}

BFloat16 operator/(int64_t lhs, BFloat16 rhs) {
    return static_cast<BFloat16>(lhs) / rhs;
}

bool operator>(BFloat16& lhs, BFloat16& rhs) {
    return static_cast<float>(lhs) > static_cast<float>(rhs);
}

bool operator<(BFloat16& lhs, BFloat16& rhs) {
    return static_cast<float>(lhs) < static_cast<float>(rhs);
}

}// namespace aethermind
