//
// Created by richard on 9/7/25.
//

#ifndef AETHERMIND_HALF_H
#define AETHERMIND_HALF_H

#include "macros.h"
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
uint32_t half_to_fp32_bits(uint16_t h);

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
float half_to_fp32_value(uint16_t h);

uint16_t half_from_fp32_value(float f);

}// namespace details

struct alignas(2) Half {
    uint16_t x;

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    Half() : x(0) {}
    constexpr Half(uint16_t bits, from_bits_t) : x(bits) {}
    Half(float value);

    operator float() const;
};

std::ostream& operator<<(std::ostream& os, const Half& value);

/// Arithmetic
Half operator+(const Half& lhs, const Half& rhs);
Half operator-(const Half& lhs, const Half& rhs);
Half operator*(const Half& lhs, const Half& rhs);
Half operator/(const Half& lhs, const Half& rhs);
Half operator-(const Half&);
Half& operator+=(Half& lhs, const Half& rhs);
Half& operator-=(Half& lhs, const Half& rhs);
Half& operator*=(Half& lhs, const Half& rhs);
Half& operator/=(Half& lhs, const Half& rhs);

/// Arithmetic with float
float operator+(Half lhs, float rhs);
float operator-(Half lhs, float rhs);
float operator*(Half lhs, float rhs);
float operator/(Half lhs, float rhs);
float operator+(float lhs, Half rhs);
float operator-(float lhs, Half rhs);
float operator*(float lhs, Half rhs);
float operator/(float lhs, Half rhs);
float& operator+=(float& lhs, const Half& rhs);
float& operator-=(float& lhs, const Half& rhs);
float& operator*=(float& lhs, const Half& rhs);
float& operator/=(float& lhs, const Half& rhs);

/// Arithmetic with double
double operator+(Half lhs, double rhs);
double operator-(Half lhs, double rhs);
double operator*(Half lhs, double rhs);
double operator/(Half lhs, double rhs);
double operator+(double lhs, Half rhs);
double operator-(double lhs, Half rhs);
double operator*(double lhs, Half rhs);
double operator/(double lhs, Half rhs);

/// Arithmetic with int
Half operator+(Half lhs, int rhs);
Half operator-(Half lhs, int rhs);
Half operator*(Half lhs, int rhs);
Half operator/(Half lhs, int rhs);
Half operator+(int lhs, Half rhs);
Half operator-(int lhs, Half rhs);
Half operator*(int lhs, Half rhs);
Half operator/(int lhs, Half rhs);

//// Arithmetic with int64_t
Half operator+(Half lhs, int64_t rhs);
Half operator-(Half lhs, int64_t rhs);
Half operator*(Half lhs, int64_t rhs);
Half operator/(Half lhs, int64_t rhs);
Half operator+(int64_t lhs, Half rhs);
Half operator-(int64_t lhs, Half rhs);
Half operator*(int64_t lhs, Half rhs);
Half operator/(int64_t lhs, Half rhs);

}// namespace aethermind

namespace std {
template<>
class numeric_limits<aethermind::Half> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr aethermind::Half min() {
        return {0x0400, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half lowest() {
        return {0xFBFF, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half max() {
        return {0x7BFF, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half epsilon() {
        return {0x1400, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half round_error() {
        return {0x3800, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half infinity() {
        return {0x7C00, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half quiet_NaN() {
        return {0x7E00, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half signaling_NaN() {
        return {0x7D00, aethermind::Half::from_bits()};
    }
    static constexpr aethermind::Half denorm_min() {
        return {0x0001, aethermind::Half::from_bits()};
    }
};
}// namespace std

#endif//AETHERMIND_HALF_H
