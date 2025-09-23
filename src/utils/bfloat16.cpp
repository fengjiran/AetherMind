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
    uint32_t rounding_bias = (x >> 16 & 1) + UINT32_C(0x7FFFF);
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

}// namespace details
}// namespace aethermind