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

    // inf or nan
    if (exponent == 0x78000000) {
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


}// namespace details
}// namespace aethermind