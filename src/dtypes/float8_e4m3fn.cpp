/// \file
/// Implementation of `Float8_e4m3fn` and E4M3FN ↔ binary32 conversion.

#include "aethermind/dtypes/float8_e4m3fn.h"

namespace aethermind {
namespace detail {

static uint32_t fp8e4m3fn_to_fp32_bits(uint8_t input) {
    // Place the E4M3FN byte in the high 8 bits of a 32-bit word so the
    // sign/exponent/mantissa fields land on familiar fp32 boundaries:
    //   bit 31     = sign
    //   bits 27-30 = 4-bit exponent
    //   bits 24-26 = 3-bit mantissa
    const uint32_t w = static_cast<uint32_t>(input) << 24;

    const uint32_t sign = w & UINT32_C(0x80000000);

    const uint32_t exponent = w & UINT32_C(0x78000000);
    const uint32_t mantissa = w & UINT32_C(0x07000000);

    // ±0: preserve sign bit, clear everything else.
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }

    // E4M3FN NaN is the unique pattern exp=1111, mantissa=111. There is no
    // infinity encoding. Map the NaN payload into an fp32 NaN by setting
    // the fp32 exponent to all-ones and shifting the 3-bit mantissa into
    // the high mantissa bits of the fp32 result.
    if (exponent == 0x78000000 && mantissa == 0x07000000) {
        return sign | 0x7F800000 | mantissa >> 4;
    }

    // Renormalize subnormals via count-leading-zeros: shifting `nonsign`
    // left by `renorm_shift` brings the implicit leading 1 of the mantissa
    // into the exponent field, turning the value into a normal binary32.
    // For already-normal inputs `__builtin_clz(nonsign)` returns 4 and
    // `renorm_shift` becomes 0 (no shift).
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;

    // Compose the fp32 result: shift mantissa into position (>> 4 places
    // E4M3FN's 3 mantissa bits at the top of fp32's 23-bit mantissa field),
    // then add the rebiased exponent. fp32 bias is 127 vs E4M3FN bias 7,
    // so the bias correction is +120 (= 0x78), minus `renorm_shift` to
    // account for the leftward shift performed on subnormals.
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

    // ±0: preserve sign bit, clear everything else.
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint8_t>(sign >> 24);
    }

    uint32_t nonsign = exponent | mantissa;

    // 0x43F00000 = 480.0f, the first fp32 magnitude not representable in
    // E4M3FN. E4M3FN's max normal is 448.0 (bit pattern 0x7E); the next
    // pattern 0x7F is reserved for NaN. Values >= 480 — and all fp32
    // inf/NaN inputs — collapse to E4M3FN NaN since the format has no
    // infinity encoding.
    if (exponent == UINT32_C(0x7F800000) || nonsign >= UINT32_C(0x43F00000)) {
        return static_cast<uint8_t>(sign >> 24 | 0x7F);
    }

    // 121 << 23 = 2^-6, the smallest E4M3FN normal. Smaller values must
    // be encoded as subnormals.
    if (nonsign < UINT32_C(121) << 23) {
        // Denormalization-via-FP-add trick: adding the magic constant
        // 2^14 (= 141 << 23) to a tiny value `t` (t < 2^-6) yields an
        // fp32 number whose low bits encode the E4M3FN subnormal mantissa
        // with correct round-to-nearest-even rounding (the fp32 hardware
        // does the rounding for us at the LSB of the fp32 mantissa, which
        // is aligned with the LSB of the E4M3FN subnormal mantissa).
        // Subtract the bit pattern of 2^14 to recover the E4M3FN mantissa
        // bits, then OR in the sign.
        uint32_t denorm_mask = UINT32_C(141) << 23;
        nonsign = fp32_to_bits(fp32_from_bits(nonsign) + fp32_from_bits(denorm_mask));
        return static_cast<uint8_t>(nonsign - denorm_mask) | static_cast<uint8_t>(sign >> 24);
    }

    // Normal path: rebias exponent fp32(127) → E4M3FN(7), truncate mantissa
    // from 23 bits to 3 bits, then round-to-nearest-even on the dropped bits.
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    uint32_t res = sign >> 24;
    res |= static_cast<uint32_t>(exp32 + 7) << 3;
    res |= mantissa >> 20;

    // Round-to-nearest-even: the 20 dropped mantissa bits split into the
    // round bit (bit 19, mask 0x00080000) and the sticky bits (bits 0..18,
    // mask 0x0007FFFF). Round up iff the round bit is set AND
    // (any sticky bit is set OR the result LSB is 1) — the standard RNE
    // rule: round half away from zero only when the result LSB would
    // otherwise be odd (tie-breaks to even).
    const uint32_t rounding_bit = mantissa & UINT32_C(0x00080000);
    const uint32_t sticky_bits = mantissa & UINT32_C(0x0007FFFF);

    if (rounding_bit && (sticky_bits || (res & 1))) {
        res += 1;
    }

    return static_cast<uint8_t>(res);
}
}// namespace detail

Float8_e4m3fn::Float8_e4m3fn(float value) : x(detail::fp8e4m3fn_from_fp32_value(value)) {}

Float8_e4m3fn::operator float() const {
    return detail::fp8e4m3fn_to_fp32_value(x);
}

bool Float8_e4m3fn::isnan() const {
    // The unique E4M3FN NaN pattern after stripping the sign bit is 0x7F
    // (exp=1111, mantissa=111). All other patterns are finite — there is
    // no infinity encoding to disambiguate from.
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

Float8_e4m3fn operator/(const Float8_e4m3fn& lhs, const Float8_e4m3fn& rhs)
        // IEEE 754 defines float division by zero as ±inf; suppress UBSan
        // which treats it as undefined behavior.
        __ubsan_ignore_float_divide_by_zero__ {
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
    return static_cast<Float8_e4m3fn>(lhs) - rhs;
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
