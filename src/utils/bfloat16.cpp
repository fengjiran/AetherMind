//
// Created by richard on 9/23/25.
//

#include "utils/bfloat16.h"

namespace aethermind {
namespace details {

float bf16_to_fp32_value(uint16_t input) {
    uint32_t tmp = input;
    tmp <<= 16;
    return fp32_from_bits(tmp);
}

uint16_t bf16_from_fp32_value(float value) {
    if (std::isnan(value)) {
        return UINT16_C(0x7FC0);
    }

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

BFloat16 operator/(const BFloat16& lhs, const BFloat16& rhs) __ubsan_ignore_float_divide_by_zero__ {
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

bool operator>(const BFloat16& lhs, const BFloat16& rhs) {
    return static_cast<float>(lhs) > static_cast<float>(rhs);
}

bool operator<(const BFloat16& lhs, const BFloat16& rhs) {
    return static_cast<float>(lhs) < static_cast<float>(rhs);
}

}// namespace aethermind
