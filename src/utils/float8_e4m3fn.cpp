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

    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint8_t>(sign >> 24);
    }

    // inf or nan
    if (exponent == UINT32_C(0x7F800000)) {
        if (mantissa == 0) {// inf(fp8 maximum)
            return static_cast<uint8_t>(sign >> 24 | 0x7E);
        }
        // nan
        return static_cast<uint8_t>(sign >> 24 | 0x7F);
    }

    // normalize the exponent from fp32 bias(127) to fp8e4m3fn bias(7)
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);
    if (exp32 < -6) {
        // The input number is smaller than 2^(-6), which is the smallest
        // fp8e4m3fn normal number, underflow to zero
        return static_cast<uint8_t>(sign >> 24);
    }

    if (exp32 > 7) {
        // overflow to max norm
        return static_cast<uint8_t>(sign >> 24 | 0x7E);
    }

    // convert to fp8 format
    uint32_t res = sign >> 24;

    // add fp8 bias (7) to exponent
    res |= static_cast<uint8_t>(exp32 + 7) << 3;

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
}// namespace aethermind