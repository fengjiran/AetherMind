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

/// Defines the Float8_e4m3fn type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
/// bias = 7
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf

namespace aethermind {

namespace details {

/*
 * Convert a 8-bit floating-point number in fp8 E4M3FN format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
float fp8e4m3fn_to_fp32_value(uint8_t input);

}

}


#endif//AETHERMIND_FLOAT8_E4M3FN_H
