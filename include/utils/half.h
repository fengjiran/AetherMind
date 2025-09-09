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

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in a bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline uint32_t half_to_fp32_bits(uint16_t h) {
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
    const uint32_t sign = w & 0x80000000U;


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
}

inline uint32_t half_to_fp32_bits_test(uint16_t h) {
    const uint32_t sign = (h & 0x8000) << 16;
    const uint32_t exponent = (h & 0x7C00) >> 10;
    const uint32_t mantissa = h & 0x03FF;
    // return __builtin_clz(mantissa);

    if (exponent == 0x1F) {
        return sign | 0x7F800000 | (mantissa << 13);
    }

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign;
        }

    }
}

inline float half_to_fp32_value(uint16_t h) {
    // 提取 FP16 的各个部分
    uint32_t sign = (h & 0x8000) << 16;    // 符号位
    uint32_t exponent = (h & 0x7C00) >> 10;// 指数位
    uint32_t mantissa = h & 0x03FF;      // 尾数位

    // 处理特殊情况
    if (exponent == 0) {
        // 零或非规约数
        if (mantissa == 0) {
            // 零
            return std::bit_cast<float>(sign);
        }
        // 非规约数 - 转换为规约形式
        // 计算前导零的数量以归一化
        int shift = 10 - __builtin_clz(mantissa);
        exponent = 127 - 14 - shift;                    // 调整指数
        mantissa = (mantissa << (shift + 1)) & 0x7FE000;// 调整尾数
    } else if (exponent == 0x1F) {
        // 无穷大或 NaN
        exponent = 0xFF;// FP32 的最大指数
    } else {
        // 规约数
        exponent = exponent + (127 - 15);// 调整指数偏置
        mantissa = mantissa << 13;       // 扩展尾数
    }

    // 组合 FP32 的各个部分
    uint32_t fp32 = sign | (exponent << 23) | mantissa;
    return std::bit_cast<float>(fp32);
}

}// namespace details
}// namespace aethermind

#endif//AETHERMIND_HALF_H
