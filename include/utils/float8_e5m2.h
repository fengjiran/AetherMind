//
// Created by 赵丹 on 2025/9/17.
//

#ifndef AETHERMIND_FLOAT8_E5M2_H
#define AETHERMIND_FLOAT8_E5M2_H

/// Defines the Float8_e5m2 type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeeee mm
/// 1 sign bit
/// 5 exponent bits
/// 2 mantissa bits
/// bias = 15
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf

#include "utils/half.h"

namespace aethermind {
namespace details {

/*
 * Convert a 8-bit floating-point number in fp8 E5M2 format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
float fp8e5m2_to_fp32_value(uint8_t input);

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
uint8_t fp8e5m2_from_fp32_value(float f);

}// namespace details
}// namespace aethermind


#endif//AETHERMIND_FLOAT8_E5M2_H
