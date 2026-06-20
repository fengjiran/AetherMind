/// \file
/// Software IEEE 754 half-precision conversion implementations.
///
/// Contains the portable integer bit-manipulation paths for binary16 ↔ binary32
/// conversion. When `X86_F16` is defined, these are replaced by hardware
/// intrinsics at the call site.

#include "utils/half.h"
#include "macros.h"
#include "utils/floating_point_utils.h"

namespace aethermind {
namespace details {

uint32_t fp16_to_fp32_bits(uint16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;

    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t exponent = w & UINT32_C(0x7C000000);
    const uint32_t mantissa = w & UINT32_C(0x03FF0000);

    // zero
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }

    // inf or nan
    if (exponent == 0x7C000000) {
        return sign | 0x7F800000 | mantissa >> 3;
    }

    // Renormalize denormalized half values into the normalized fp32 range.
    // __builtin_clz finds the position of the leading 1 bit; we shift the
    // mantissa into the normalized position and adjust the exponent bias
    // accordingly. The `> 5` guard accounts for the 5-bit exponent field
    // (fp32 mantissa is 23 bits vs half's 10).
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    return sign | (nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23);
}

float fp16_to_fp32_value(uint16_t h) {
    return fp32_from_bits(fp16_to_fp32_bits(h));
}

uint16_t fp16_from_fp32_value(float f) {
    uint32_t x = fp32_to_bits(f);
    const uint32_t sign = x & UINT32_C(0x80000000);
    const uint32_t exponent = x & UINT32_C(0x7F800000);
    const uint32_t mantissa = x & UINT32_C(0x007FFFFF);

    // zero case
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint16_t>(sign >> 16);
    }

    // inf and nan case
    if (exponent == UINT32_C(0x7F800000)) {
        // inf case: set exponent to max and mantissa to zero
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign >> 16 | 0x7C00);
        }

        // NaN: canonicalize to quiet NaN (0x7E00), discarding the payload.
        // IEEE 754 requires at least one NaN; preserving the payload would
        // increase implementation complexity with no known use case.
        return static_cast<uint16_t>(sign >> 16 | 0x7E00);
    }

    // normalize the exponent from fp32 bias(127) to fp16 bias(15)
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    // Handle values too small to represent even as half denormals: flush to zero.
    if (exp32 < -14) {
        return static_cast<uint16_t>(sign >> 16);
    }

    if (exp32 > 15) {
        // overflow to inf
        return static_cast<uint16_t>(sign >> 16 | 0x7C00);
    }

    // convert to fp16 format
    uint32_t res = sign >> 16;

    // add fp16 bias (15) to exponent
    res |= static_cast<uint32_t>(exp32 + 15) << 10;

    // add mantissa (truncated from fp32; rounding is handled below)
    res |= mantissa >> 13;

    // IEEE 754 round-to-nearest-even.
    // rounding_bit is the first bit lost in truncation (bit 12 of mantissa).
    // sticky_bits are the remaining lost bits (bits 0-11).
    // Round up when rounding_bit is set AND (any sticky bits are set OR the
    // result is odd) — the `res & 1` check implements the "even" tie-break.
    const uint32_t rounding_bit = mantissa & UINT32_C(0x00001000);
    const uint32_t sticky_bits = mantissa & UINT32_C(0x00000FFF);

    if (rounding_bit && (sticky_bits || (res & 1))) {
        res += 1;
    }

    return static_cast<uint16_t>(res);
}
}// namespace details

Half::Half(float value) : x(details::fp16_from_fp32_value(value)) {}

Half::operator float() const {
    return details::fp16_to_fp32_value(x);
}

std::ostream& operator<<(std::ostream& os, const Half& value) {
    os << static_cast<float>(value);
    return os;
}

Half operator+(const Half& lhs, const Half& rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

Half operator-(const Half& lhs, const Half& rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

Half operator*(const Half& lhs, const Half& rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

Half operator/(const Half& lhs, const Half& rhs)
        // IEEE 754 defines float division by zero as ±inf; suppress UBSan
        // which treats it as undefined behavior.
        __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

Half operator-(const Half& a) {
    return -static_cast<float>(a);
}

Half& operator+=(Half& lhs, const Half& rhs) {
    lhs = lhs + rhs;
    return lhs;
}

Half& operator-=(Half& lhs, const Half& rhs) {
    lhs = lhs - rhs;
    return lhs;
}

Half& operator*=(Half& lhs, const Half& rhs) {
    lhs = lhs * rhs;
    return lhs;
}

Half& operator/=(Half& lhs, const Half& rhs) {
    lhs = lhs / rhs;
    return lhs;
}

float operator+(Half lhs, float rhs) {
    return static_cast<float>(lhs) + rhs;
}

float operator-(Half lhs, float rhs) {
    return static_cast<float>(lhs) - rhs;
}

float operator*(Half lhs, float rhs) {
    return static_cast<float>(lhs) * rhs;
}

float operator/(Half lhs, float rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<float>(lhs) / rhs;
}

float operator+(float lhs, Half rhs) {
    return lhs + static_cast<float>(rhs);
}

float operator-(float lhs, Half rhs) {
    return lhs - static_cast<float>(rhs);
}

float operator*(float lhs, Half rhs) {
    return lhs * static_cast<float>(rhs);
}

float operator/(float lhs, Half rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<float>(rhs);
}

float& operator+=(float& lhs, const Half& rhs) {
    return lhs += static_cast<float>(rhs);
}

float& operator-=(float& lhs, const Half& rhs) {
    return lhs -= static_cast<float>(rhs);
}

float& operator*=(float& lhs, const Half& rhs) {
    return lhs *= static_cast<float>(rhs);
}

float& operator/=(float& lhs, const Half& rhs) {
    return lhs /= static_cast<float>(rhs);
}

double operator+(Half lhs, double rhs) {
    return static_cast<double>(lhs) + rhs;
}

double operator-(Half lhs, double rhs) {
    return static_cast<double>(lhs) - rhs;
}

double operator*(Half lhs, double rhs) {
    return static_cast<double>(lhs) * rhs;
}

double operator/(Half lhs, double rhs) __ubsan_ignore_float_divide_by_zero__ {
    return static_cast<double>(lhs) / rhs;
}

double operator+(double lhs, Half rhs) {
    return lhs + static_cast<double>(rhs);
}

double operator-(double lhs, Half rhs) {
    return lhs - static_cast<double>(rhs);
}

double operator*(double lhs, Half rhs) {
    return lhs * static_cast<double>(rhs);
}

double operator/(double lhs, Half rhs) __ubsan_ignore_float_divide_by_zero__ {
    return lhs / static_cast<double>(rhs);
}

Half operator+(Half lhs, int rhs) {
    return lhs + static_cast<Half>(rhs);
}

Half operator-(Half lhs, int rhs) {
    return lhs - static_cast<Half>(rhs);
}

Half operator*(Half lhs, int rhs) {
    return lhs * static_cast<Half>(rhs);
}

Half operator/(Half lhs, int rhs) {
    return lhs / static_cast<Half>(rhs);
}

Half operator+(int lhs, Half rhs) {
    return static_cast<Half>(lhs) + rhs;
}

Half operator-(int lhs, Half rhs) {
    return static_cast<Half>(lhs) - rhs;
}

Half operator*(int lhs, Half rhs) {
    return static_cast<Half>(lhs) * rhs;
}

Half operator/(int lhs, Half rhs) {
    return static_cast<Half>(lhs) / rhs;
}

Half operator+(Half lhs, int64_t rhs) {
    return lhs + static_cast<Half>(rhs);
}

Half operator-(Half lhs, int64_t rhs) {
    return lhs - static_cast<Half>(rhs);
}

Half operator*(Half lhs, int64_t rhs) {
    return lhs * static_cast<Half>(rhs);
}

Half operator/(Half lhs, int64_t rhs) {
    return lhs / static_cast<Half>(rhs);
}

Half operator+(int64_t lhs, Half rhs) {
    return static_cast<Half>(lhs) + rhs;
}

Half operator-(int64_t lhs, Half rhs) {
    return static_cast<Half>(lhs) - rhs;
}

Half operator*(int64_t lhs, Half rhs) {
    return static_cast<Half>(lhs) * rhs;
}

Half operator/(int64_t lhs, Half rhs) {
    return static_cast<Half>(lhs) / rhs;
}

}// namespace aethermind