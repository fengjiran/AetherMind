//
// Created by richard on 9/7/25.
//

#ifndef AETHERMIND_HALF_H
#define AETHERMIND_HALF_H

#include "utils/floating_point_utils.h"

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>


#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#if defined(__aarch64__) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif


#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
        defined(_M_IX86)
#if defined(__F16C__) &&                                   \
        !(defined(__CUDA_ARCH__) || defined(__CUDACC__) || \
          defined(__HIP_DEVICE_COMPILE__))
#define X86_F16 1
#include <immintrin.h>// import conversion ops from f16cintrin.h
#endif                // defined(__F16C__) && !(defined(__CUDA_ARCH__) || defined(__CUDACC__) \
                      // || defined(__HIP_DEVICE_COMPILE__))
#endif                // __x86_64__ || _M_X64 || __i386 || _M_IX86
#endif                // __GNUC__ || __clang__
namespace aethermind {
namespace details {

inline uint32_t half_to_fp32_bits_benchmark(uint16_t h) {
    /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
    const uint32_t w = static_cast<uint32_t>(h) << 16;

    /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
    const uint32_t sign = w & UINT32_C(0x80000000);

    /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
    const uint32_t nonsign = w & 0x7FFFFFFFU;

    /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalized, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;

    /*
   * If half-precision number has exponent of 15, the addition overflows
   * it into bit 31, and the subsequent shift turns the high 9 bits
   * into 1. Thus, inf_nan_mask == 0x7F800000 if the half-precision number
   * had exponent of 15 (i.e. was NaN or infinity) 0x00000000 otherwise
   */
    const int32_t inf_nan_mask = (static_cast<int32_t>(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);

    /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31
   * into 1. Otherwise, bit 31 remains 0. The signed shift right by 31
   * broadcasts bit 31 into all bits of the zero_mask. Thus zero_mask ==
   * 0xFFFFFFFF if the half-precision number was zero (+0.0h or -0.0h)
   * 0x00000000 otherwise
   */
    const int32_t zero_mask = (int32_t) (nonsign - 1) >> 31;

    /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input
   * was denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally)
   * becomes an 8-bit field and 10-bit mantissa shifts into the 10 high
   * bits of the 23-bit mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF
   * for half-precision number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to
   * account for renormalization. As renorm_shift is less than 0x70, this
   * can be combined with step 3.
   * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the
   * input was NaN or infinity.
   * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent
   * into zero if the input was zero.
   * 7. Combine with the sign of the input number.
   */
    return sign |
           ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
             inf_nan_mask) &
            ~zero_mask);
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in a bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline uint32_t half_to_fp32_bits(uint16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;

    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
    const uint32_t exponent = w & UINT32_C(0x7C000000);
    const uint32_t mantissa = w & UINT32_C(0x03FF0000);

    // inf or nan
    if (exponent == 0x7C000000) {
        return sign | 0x7F800000 | mantissa >> 3;
    }

    // zero
    if (exponent == 0 && mantissa == 0) {
        return sign;
    }

    uint32_t renorm_shift = __builtin_clz(nonsign);
    renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
    return sign | (nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23);
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
inline float half_to_fp32_value(uint16_t h) {
    return fp32_from_bits(half_to_fp32_bits(h));
}

inline uint16_t half_from_fp32_value(float f) {
    uint32_t x = fp32_to_bits(f);
    const uint32_t sign = x & UINT32_C(0x80000000);
    const uint32_t exponent = x & UINT32_C(0x7F800000);
    const uint32_t mantissa = x & UINT32_C(0x007FFFFF);

    // zero case
    if (exponent == 0 && mantissa == 0) {
        return static_cast<uint16_t>(sign >> 16);
    }

    // inf and nan case
    if (exponent == UINT32_C(0x7F800000)) {
        // inf case: set exponent to max and mantissa to zero
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign >> 16 | 0x7C00);
        }

        // nan case: preserve mantissa bits and set exponent to max
        return static_cast<uint16_t>(sign >> 16 | 0x7C00 | mantissa >> 13);
    }

    // normalize the exponent from fp32 bias(127) to fp16 bias(15)
    auto exp32 = static_cast<int32_t>((exponent >> 23) - 127);

    // handle denorm numbers(exponent underflow)
    if (exp32 < -14) {
        // underflow to zero
        return static_cast<uint16_t>(sign >> 16);
    }

    if (exp32 > 15) {
        // overflow to inf
        return static_cast<uint16_t>(sign >> 16 | 0x7C00);
    }

    // convert to fp16 format
    uint32_t res = sign >> 16;

    // add fp16 bias (15) to exponent
    res |= static_cast<uint32_t>(exp32 + 15) << 10;

    // add mantissa (round to nearest even)
    res |= mantissa >> 13;

    // handle rounding
    const uint32_t rounding_bit = mantissa & UINT32_C(0x00001000);
    const uint32_t sticky_bits = mantissa & UINT32_C(0x00000FFF);

    if (rounding_bit && (sticky_bits || (res & 1))) {
        res += 1;
    }

    return static_cast<uint16_t>(res);
}

}// namespace details
}// namespace aethermind

#endif//AETHERMIND_HALF_H
