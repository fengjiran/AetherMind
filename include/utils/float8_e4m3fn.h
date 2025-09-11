//
// Created by richard on 9/6/25.
//

#ifndef AETHERMIND_FLOAT8_E4M3FN_H
#define AETHERMIND_FLOAT8_E4M3FN_H

#include <cmath>
#include <cstdint>
#include <climits>
#include <iostream>

#include "utils/floating_point_utils.h"

namespace aethermind {

namespace details {

/*
 * Convert a 8-bit floating-point number in fp8 E4M3FN format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
// inline float fp8e4m3fn_to_fp32_value(uint8_t input) {
//
// }

}

}


#endif//AETHERMIND_FLOAT8_E4M3FN_H
