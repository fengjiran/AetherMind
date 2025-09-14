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

    // // inf or nan
    // if (exponent == 0x78000000) {
    //     return sign | 0x7F800000 | mantissa >> 4;
    // }

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

    uint32_t nonsign = x & UINT32_C(0x7FFFFFFF);

    // convert inf, nan and greater than fp8e4m3fn max(480.0f) in fp32 to fp8e4m3fn max
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

    // resulting mantissa is odd
    uint8_t mant_odd = (nonsign >> 20) & 1;

    // update exponent, rounding bias part 1
    nonsign += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFF;

    // rounding bias part 2
    nonsign += mant_odd;

    auto res = static_cast<uint8_t>(nonsign >> 20);

    res |= static_cast<uint8_t>(sign >> 24);

    return res;
}


}// namespace details
}// namespace aethermind