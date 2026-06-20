/// \file
/// Brain floating-point (bfloat16) type.
///
/// Provides `BFloat16` — a C++ type wrapping a 16-bit bfloat16 bit pattern —
/// and conversion functions between bfloat16 and IEEE 754 binary32. The
/// bfloat16 format (1 sign | 8 exponent | 7 mantissa) shares its exponent
/// range with binary32 but has a 7-bit mantissa, so conversion to/from float
/// is a bit truncation/extension at the mantissa boundary.

#ifndef AETHERMIND_DTYPES_BFLOAT16_H
#define AETHERMIND_DTYPES_BFLOAT16_H

#include "aethermind/dtypes/detail/floating_point_bits.h"
#include "aethermind/dtypes/half.h"

#include <cmath>
#include <cstdint>
#include <iosfwd>
#include <ostream>

namespace aethermind {
namespace details {

/// Converts a bfloat16 bit pattern to an IEEE 754 binary32 `float`.
///
/// bfloat16 occupies the high 16 bits of a binary32; the conversion left-shifts
/// the input by 16 and reinterprets as `float`. Bit-exact: NaN payloads, sign,
/// and exponent are preserved.
float bf16_to_fp32_value(uint16_t input);

/// Converts an IEEE 754 binary32 `float` to a bfloat16 bit pattern.
///
/// Uses round-to-nearest-even on the truncated 16 mantissa LSBs. NaN inputs
/// are canonicalized to the quiet NaN bit pattern `0x7FC0`.
uint16_t bf16_from_fp32_value(float);

}// namespace details

/// Brain floating-point (bfloat16) value.
///
/// Stores the raw 16-bit bfloat16 bit pattern in `x`. The format is
/// 1 sign | 8 exponent | 7 mantissa, which shares the exponent range of
/// binary32 but has lower precision. Arithmetic and comparisons go through
/// implicit conversion to `float` and follow IEEE 754 binary32 rounding and
/// NaN semantics. Use `from_bits()` to construct from a raw bit pattern
/// without conversion.
struct alignas(2) BFloat16 {
    /// bfloat16 bit pattern: 1 sign | 8 exponent | 7 mantissa.
    uint16_t x;

    /// Tag type for constructing a `BFloat16` from raw bits without conversion.
    struct from_bits_t {};
    static constexpr from_bits_t from_bits() {
        return {};
    }

    BFloat16() : x(0) {}
    /// Constructs from raw bfloat16 bits; no floating-point conversion.
    constexpr BFloat16(uint16_t bits, from_bits_t) : x(bits) {}
    BFloat16(float);       // NOLINT
    operator float() const;// NOLINT
};

std::ostream& operator<<(std::ostream& os, const BFloat16& value);

/// Arithmetic
BFloat16 operator+(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator-(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator*(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator/(const BFloat16& lhs, const BFloat16& rhs);
BFloat16 operator-(const BFloat16&);
BFloat16& operator+=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator-=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator*=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator/=(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator|(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator^(BFloat16& lhs, const BFloat16& rhs);
BFloat16& operator&(BFloat16& lhs, const BFloat16& rhs);

/// Arithmetic with float
float operator+(BFloat16 lhs, float rhs);
float operator-(BFloat16 lhs, float rhs);
float operator*(BFloat16 lhs, float rhs);
float operator/(BFloat16 lhs, float rhs);
float operator+(float lhs, BFloat16 rhs);
float operator-(float lhs, BFloat16 rhs);
float operator*(float lhs, BFloat16 rhs);
float operator/(float lhs, BFloat16 rhs);
float& operator+=(float& lhs, const BFloat16& rhs);
float& operator-=(float& lhs, const BFloat16& rhs);
float& operator*=(float& lhs, const BFloat16& rhs);
float& operator/=(float& lhs, const BFloat16& rhs);

/// Arithmetic with double
double operator+(BFloat16 lhs, double rhs);
double operator-(BFloat16 lhs, double rhs);
double operator*(BFloat16 lhs, double rhs);
double operator/(BFloat16 lhs, double rhs);
double operator+(double lhs, BFloat16 rhs);
double operator-(double lhs, BFloat16 rhs);
double operator*(double lhs, BFloat16 rhs);
double operator/(double lhs, BFloat16 rhs);

/// Arithmetic with int
BFloat16 operator+(BFloat16 lhs, int rhs);
BFloat16 operator-(BFloat16 lhs, int rhs);
BFloat16 operator*(BFloat16 lhs, int rhs);
BFloat16 operator/(BFloat16 lhs, int rhs);
BFloat16 operator+(int lhs, BFloat16 rhs);
BFloat16 operator-(int lhs, BFloat16 rhs);
BFloat16 operator*(int lhs, BFloat16 rhs);
BFloat16 operator/(int lhs, BFloat16 rhs);

/// Arithmetic with int64_t
BFloat16 operator+(BFloat16 lhs, int64_t rhs);
BFloat16 operator-(BFloat16 lhs, int64_t rhs);
BFloat16 operator*(BFloat16 lhs, int64_t rhs);
BFloat16 operator/(BFloat16 lhs, int64_t rhs);
BFloat16 operator+(int64_t lhs, BFloat16 rhs);
BFloat16 operator-(int64_t lhs, BFloat16 rhs);
BFloat16 operator*(int64_t lhs, BFloat16 rhs);
BFloat16 operator/(int64_t lhs, BFloat16 rhs);

bool operator>(BFloat16& lhs, BFloat16& rhs);
bool operator<(BFloat16& lhs, BFloat16& rhs);

/// Trait identifying reduced-precision floating-point types (`Half`,
/// `BFloat16`). Used to gate `std::` math overloads that route through
/// `float` for these types.
template<typename T>
struct is_reduced_floating_point : std::false_type {};

template<>
struct is_reduced_floating_point<Half> : std::true_type {};

template<>
struct is_reduced_floating_point<BFloat16> : std::true_type {};

template<typename T>
constexpr bool is_reduced_floating_point_v = is_reduced_floating_point<T>::value;

}// namespace aethermind

namespace std {

/// std::numeric_limits specialization for bfloat16.
///
/// Bit constants follow the bfloat16 format (1 sign | 8 exponent | 7 mantissa,
/// bias 127). `is_iec559` is false because bfloat16 is not an IEEE 754
/// standard format, even though it uses IEEE-like semantics.
template<>
struct numeric_limits<aethermind::BFloat16> {
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr aethermind::BFloat16 min() {
        return {0x0080, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 lowest() {
        return {0xFF7F, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 max() {
        return {0x7F7F, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 epsilon() {
        return {0x3C00, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 round_error() {
        return {0x3F00, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 infinity() {
        return {0x7F80, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 quiet_NaN() {
        return {0x7FC0, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 signaling_NaN() {
        // Need exponent=0xFF, mantissa!=0, mantissa MSB=0. 0x7F80 is +infinity.
        return {0x7F81, aethermind::BFloat16::from_bits()};
    }

    static constexpr aethermind::BFloat16 denorm_min() {
        return {0x0001, aethermind::BFloat16::from_bits()};
    }
};

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T acos(T a) {
    return std::acos(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T asin(T a) {
    return std::asin(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T atan(T a) {
    return std::atan(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T erf(T a) {
    return std::erf(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T erfc(T a) {
    return std::erfc(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T exp(T a) {
    return std::exp(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T expm1(T a) {
    return std::expm1(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
bool isfinite(T a) {
    return std::isfinite(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T log(T a) {
    return std::log(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T log10(T a) {
    return std::log10(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T log2(T a) {
    return std::log2(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T log1p(T a) {
    return std::log1p(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T ceil(T a) {
    return std::ceil(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T cos(T a) {
    return std::cos(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T sin(T a) {
    return std::sin(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T sinh(T a) {
    return std::sinh(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T cosh(T a) {
    return std::cosh(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T tan(T a) {
    return std::tan(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T tanh(T a) {
    return std::tanh(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T floor(T a) {
    return std::floor(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T nearbyint(T a) {
    return std::nearbyint(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T trunc(T a) {
    return std::trunc(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T lgamma(T a) {
    return std::lgamma(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T sqrt(T a) {
    return std::sqrt(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T rsqrt(T a) {
    return 1.0 / std::sqrt(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T abs(T a) {
    return std::abs(static_cast<float>(a));
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T pow(T a, double b) {
    return std::pow(static_cast<float>(a), b);
}

template<typename T, typename = std::enable_if_t<aethermind::is_reduced_floating_point_v<T>>>
T fmod(T a, T b) {
    return std::fmod(static_cast<float>(a), static_cast<float>(b));
}

}// namespace std

#endif// AETHERMIND_DTYPES_BFLOAT16_H
