/// \file
/// Implementation of `Float8_e5m2` and E5M2 ↔ binary32 conversion.
#include "utils/float8_e5m2.h"
#include "macros.h"
#include "utils/floating_point_utils.h"

namespace aethermind {
namespace {

// Converts an E5M2 bit pattern to an IEEE 754 binary32 bit pattern.
// E5M2 layout (1 sign | 5 exponent (bias 15) | 2 mantissa) is bit-equivalent
// to binary16 with the low 8 mantissa bits zeroed. Left-shifting by 24 puts
// the E5M2 byte at the same bit positions a binary16 occupies after a 16-bit
// left shift, so the binary16 algorithm applies verbatim with identical
// masks and shift counts.
uint32_t fp8e5m2_to_fp32_bits(uint8_t input) {
    const uint32_t w = static_cast<uint32_t>(input) << 24;

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

    // Renormalize subnormal values into the binary32 normal range.
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    return sign | (nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23);
}

}// namespace

namespace details {

float fp8e5m2_to_fp32_value(uint8_t input) {
    return fp32_from_bits(fp8e5m2_to_fp32_bits(input));
}

uint8_t fp8e5m2_from_fp32_value(float f) {
    uint32_t x = fp32_to_bits(f);
    const uint32_t sign = x & UINT32_C(0x80000000);
    const uint32_t exponent = x & UINT32_C(0x7F800000);
    const uint32_t mantissa = x & UINT32_C(0x007FFFFF);

    // ±0: preserve sign bit, clear everything else.
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint8_t>(sign >> 24);
    }

    // Inf and NaN have fp32 exponent = all-ones. E5M2 encodes them as
    // exp=11111 with mantissa=00 (inf, 0x7C) or mantissa≠00 (NaN, 0x7E).
    if (exponent == UINT32_C(0x7F800000)) {
        if (mantissa == 0) {
            return static_cast<uint8_t>(sign >> 24 | 0x7C);
        }
        return static_cast<uint8_t>(sign >> 24 | 0x7E);
    }

    // 0x47800000 = 2^16, the smallest fp32 magnitude that exceeds E5M2's
    // representable range. Saturate to ±inf for finite overflow; preserve
    // NaN bit pattern (sign | 0x7F) for fp32 NaNs that slipped past the
    // exponent==0xFF check above (they cannot — but the branch is kept
    // defensive).
    uint32_t nonsign = exponent | mantissa;
    if (nonsign >= UINT32_C(0x47800000)) {
        return nonsign > UINT32_C(0x7F800000) ? static_cast<uint8_t>(sign >> 24 | 0x7F) : static_cast<uint8_t>(sign >> 24 | 0x7C);
    }

    // 113 << 23 = 2^-14, the smallest E5M2 normal. Smaller values must be
    // encoded as subnormals.
    if (nonsign < UINT32_C(113) << 23) {
        // Denormalization-via-FP-add trick: adding the magic constant
        // 2^7 (134 << 23) to a tiny value `t` (t < 2^-14) yields a fp32
        // number whose low bits encode the E5M2 subnormal mantissa with
        // correct round-to-nearest-even rounding (the fp32 hardware does
        // the rounding for us at the LSB of the fp32 mantissa, which is
        // aligned with the LSB of the E5M2 subnormal mantissa). Subtract
        // the bit pattern of 2^7 to recover the E5M2 mantissa bits, then
        // OR in the sign.
        uint32_t denorm_mask = UINT32_C(134) << 23;
        nonsign = fp32_to_bits(fp32_from_bits(nonsign) + fp32_from_bits(denorm_mask));
        return static_cast<uint8_t>(nonsign - denorm_mask) | static_cast<uint8_t>(sign >> 24);
    }

    // Normal path: rebias exponent fp32(127) → E5M2(15), truncate mantissa
    // from 23 bits to 2 bits, then round-to-nearest-even on the dropped bits.
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    uint32_t res = sign >> 24;
    res |= static_cast<uint32_t>(exp32 + 15) << 2;
    res |= mantissa >> 21;

    // Round-to-nearest-even: the 21 dropped mantissa bits split into the
    // round bit (bit 20, mask 0x00100000) and the sticky bits (bits 0..19,
    // mask 0x000FFFFF). Round up iff the round bit is set AND
    // (any sticky bit is set OR the result LSB is 1) — this is the standard
    // RNE rule: round half away from zero only when the result LSB would
    // otherwise be odd (tie-breaks to even).
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
    // exp=11111, mantissa=00 (after masking out the sign bit).
    return (x & 0x7F) == 0x7C;
}

bool Float8_e5m2::isnan() const {
    // exp=11111, mantissa≠00. Bit patterns 0x7D, 0x7E, 0x7F (and signed
    // counterparts) all qualify; comparing against 0x7C as a strict upper
    // bound captures all three with a single comparison.
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

Float8_e5m2 operator/(const Float8_e5m2& lhs, const Float8_e5m2& rhs)
        // IEEE 754 defines float division by zero as ±inf; suppress UBSan
        // which treats it as undefined behavior.
        __ubsan_ignore_float_divide_by_zero__ {
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
    return static_cast<float>(lhs) - rhs;
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