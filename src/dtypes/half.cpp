/// \file
/// Software IEEE 754 half-precision conversion implementations.
///
/// Contains the portable integer bit-manipulation paths for binary16 ↔ binary32
/// conversion. When `X86_F16` is defined, these are replaced by hardware
/// intrinsics at the call site.
#include "aethermind/dtypes/half.h"
#include "aethermind/dtypes/detail/floating_point_bits.h"
#include "macros.h"

namespace aethermind {
namespace {

// Converts an IEEE half-precision value bit pattern to an IEEE single-precision
// bit pattern. Implemented with integer bit manipulation only; no floating-point
// operations.
uint32_t fp16_to_fp32_bits(uint16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & 0x80000000U;
    const uint32_t exponent = w & 0x7C000000U;
    const uint32_t mantissa = w & 0x03FF0000U;

    // zero
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }

    // clang-format off
    // inf or nan
    if (exponent == 0x7C000000U) AM_UNLIKELY {
        return sign | 0x7F800000U | (mantissa >> 3);
    }
    // clang-format on

    // Renormalize denormalized half values into the normalized fp32 range.
    // std::countl_zero finds the position of the leading 1 bit; we shift the
    // mantissa into the normalized position and adjust the exponent bias
    // accordingly. The `> 5` guard accounts for the 5-bit exponent field
    // (fp32 mantissa is 23 bits vs half's 10).
    const uint32_t nonsign = w & 0x7FFFFFFFU;
    // Zero, inf, and NaN cases returned above, so nonsign is non-zero here.
    uint32_t renorm_shift = std::countl_zero(nonsign);
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    return sign | ((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23));
}

// Converts an IEEE single-precision `float` to an IEEE half-precision value
// (bit representation). Implemented with integer bit manipulation only; no
// floating-point operations. Rounding mode is round-to-nearest-even.
uint16_t fp16_from_fp32_value(float f) {
    const uint32_t x = detail::fp32_to_bits(f);
    const uint32_t sign = x & 0x80000000U;
    const uint32_t exponent = x & 0x7F800000U;
    const uint32_t mantissa = x & 0x007FFFFFU;

    // zero case
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint16_t>(sign >> 16);
    }

    // inf and nan case
    if (exponent == 0x7F800000U) {
        // inf case: set exponent to max and mantissa to zero
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign >> 16 | 0x7C00U);
        }

        // NaN: preserve the top 10 mantissa bits of the binary32 payload
        // and force the quiet-NaN bit so we never propagate a signaling NaN.
        // If the payload rounds to zero, substitute the smallest non-zero
        // quiet payload to keep the result a NaN.
        auto nan_payload = static_cast<uint16_t>(mantissa >> 13);
        if (nan_payload == 0) {
            nan_payload = 0x0200U;
        } else {
            nan_payload |= 0x0200U;
        }
        return static_cast<uint16_t>(sign >> 16 | 0x7C00U | nan_payload);
    }

    // normalize the exponent from fp32 bias(127) to fp16 bias(15)
    const auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    // We do not support FP16 denormals. Values smaller than the minimum
    // normal FP16 (exp32 < -14) are flushed to zero.
    if (exp32 < -14) {
        return static_cast<uint16_t>(sign >> 16);
    }

    if (exp32 > 15) {
        // overflow to inf
        return static_cast<uint16_t>(sign >> 16 | 0x7C00U);
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
    const uint32_t rounding_bit = mantissa & 0x00001000U;
    const uint32_t sticky_bits = mantissa & 0x00000FFFU;

    if (rounding_bit && (sticky_bits || (res & 1))) {
        res += 1;
    }

    return static_cast<uint16_t>(res);
}

// Converts an IEEE half-precision bit pattern to an IEEE single-precision
// `float`. Uses int/float bitcasts; makes no assumption about the current
// rounding mode and performs no operations on denormals.
float fp16_to_fp32_value(uint16_t h) {
    return detail::fp32_from_bits(fp16_to_fp32_bits(h));
}

}// namespace

namespace detail {

uint32_t fp16_to_fp32_bits_for_testing(uint16_t h) {
    return fp16_to_fp32_bits(h);
}

float fp16_to_fp32_value_for_testing(uint16_t h) {
    return fp16_to_fp32_value(h);
}

uint16_t fp16_from_fp32_value_for_testing(float f) {
    return fp16_from_fp32_value(f);
}

}// namespace detail

Half::Half(float value) : x(fp16_from_fp32_value(value)) {}

Half::operator float() const {
    return fp16_to_fp32_value(x);
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

float operator+(Half lhs, int rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

float operator-(Half lhs, int rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

float operator*(Half lhs, int rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

float operator/(Half lhs, int rhs) {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

float operator+(int lhs, Half rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

float operator-(int lhs, Half rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

float operator*(int lhs, Half rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

float operator/(int lhs, Half rhs) {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

float operator+(Half lhs, int64_t rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

float operator-(Half lhs, int64_t rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

float operator*(Half lhs, int64_t rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

float operator/(Half lhs, int64_t rhs) {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

float operator+(int64_t lhs, Half rhs) {
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

float operator-(int64_t lhs, Half rhs) {
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

float operator*(int64_t lhs, Half rhs) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

float operator/(int64_t lhs, Half rhs) {
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

}// namespace aethermind
