/// Numeric type casting and overflow-checking utilities.
///
/// Provides constexpr sign/overflow predicates and the `cast` /
/// `check_and_cast` families used by tensor dtype conversions.

#ifndef AETHERMIND_DTYPES_CAST_H
#define AETHERMIND_DTYPES_CAST_H

#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/complex.h"
#include "aethermind/dtypes/float8_e4m3fn.h"
#include "aethermind/dtypes/float8_e5m2.h"
#include "error.h"
#include "macros.h"

namespace aethermind {

// Unsigned case: never negative.
template<typename T>
constexpr bool is_negative(const T&, std::true_type) {
    return false;
}

// Signed case: compare against zero.
template<typename T>
constexpr bool is_negative(const T& x, std::false_type) {
    return x < T(0);
}

/// Returns true if x < 0.
///
/// @note Fails for unsigned custom types that do not specialize
///       std::is_unsigned; such types fall through to the signed path.
template<typename T>
constexpr bool is_negative(const T& x) {
    return is_negative(x, std::is_unsigned<T>());
}

/// Returns true when a and b have opposite signedness (exactly one is negative).
template<typename T, typename U>
constexpr bool signs_differ(const T& a, const U& b) {
    return is_negative(a) != is_negative(b);
}

// Unsigned case: 0 if zero, 1 otherwise.
template<typename T>
constexpr int signum(const T& x, std::true_type) {
    return T(0) < x;
}

// Signed case: -1, 0, or 1.
template<typename T>
constexpr int signum(const T& x, std::false_type) {
    return (T(0) < x) - (x < T(0));
}

/// Returns the sign of x as -1, 0, or 1.
///
/// @note Fails for unsigned custom types that do not specialize
///       std::is_unsigned; such types fall through to the signed path.
template<typename T>
constexpr int signum(const T& x) {
    return signum(x, std::is_unsigned<T>());
}

/// Returns true if x exceeds the maximum value representable by Limit.
template<typename Limit, typename T>
constexpr bool greater_than_max(const T& x) {
    constexpr bool can_overflow = std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
    return can_overflow && x > std::numeric_limits<Limit>::max();
}

// Both signed: standard comparison against lowest().
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& x,
        std::false_type /*limit_is_unsigned*/,
        std::false_type /*x_is_unsigned*/) {
    return x < std::numeric_limits<Limit>::lowest();
}

// Limit is signed but x is unsigned: x cannot be below lowest.
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& /*x*/,
        std::false_type /*limit_is_unsigned*/,
        std::true_type /*x_is_unsigned*/) {
    return false;
}

// Limit is unsigned (lower bound is zero): compare against T(0).
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& x,
        std::true_type /*limit_is_unsigned*/,
        std::false_type /*x_is_unsigned*/) {
    return x < T(0);
}

// Both unsigned: x cannot be below zero.
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& /*x*/,
        std::true_type /*limit_is_unsigned*/,
        std::true_type /*x_is_unsigned*/) {
    return false;
}

/// Returns true if x is below the lowest value representable by Limit.
///
/// @note Fails for unsigned custom types that do not specialize
///       std::is_unsigned; Half is a known counterexample.
template<typename Limit, typename T>
constexpr bool less_than_lowest(const T& x) {
    return less_than_lowest<Limit>(x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

/// bool converts to any target type without overflow.
template<typename From, typename To,
         std::enable_if_t<std::is_same_v<From, bool>>* = nullptr>
bool is_cast_overflow(From, AM_MAYBE_UNUSED bool strict_unsigned = false) {
    return false;
}

/// Checks whether src overflows when cast to To.
///
/// @param strict_unsigned when true, a negative src is rejected even for an
///        unsigned To; otherwise only the magnitude is checked
/// @return true if the conversion would overflow
template<typename From, typename To,
         std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>>* = nullptr>
bool is_cast_overflow(From src, bool strict_unsigned = false) {
    using Limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if constexpr (!Limit::is_signed && std::numeric_limits<From>::is_signed) {
        if (!strict_unsigned) {
            return greater_than_max<To>(src) ||
                   (is_negative(src) && -static_cast<uint64_t>(src) > static_cast<uint64_t>(Limit::max()));
        }
    }

    return greater_than_max<To>(src) || less_than_lowest<To>(src);
}

/// Checks whether a floating-point src overflows when cast to To.
template<typename From, typename To,
         std::enable_if_t<std::is_floating_point_v<From>>* = nullptr>
bool is_cast_overflow(From src, AM_MAYBE_UNUSED bool strict_unsigned = false) {
    using Limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if (Limit::has_infinity && std::isinf(static_cast<double>(src))) {
        return false;
    }

    if (!Limit::has_quiet_NaN && src != src) {
        return true;
    }
    return src < Limit::lowest() || src > Limit::max();
}

/// Checks whether a complex src overflows when cast to To.
template<typename From, typename To, std::enable_if_t<is_complex_v<From>>* = nullptr>
bool is_cast_overflow(From src, bool strict_unsigned = false) {
    if (!is_complex_v<To> && src.imag() != 0) {
        return true;
    }

    using from_type = From::value_type;
    using to_type = scalar_value_type<To>::type;

    return is_cast_overflow<from_type, to_type>(src.real(), strict_unsigned) ||
           is_cast_overflow<from_type, to_type>(src.imag(), strict_unsigned);
}

// True when casting complex->non-complex, so only the real part is used.
template<typename From, typename To>
constexpr static bool only_need_real = is_complex_v<From> && !is_complex_v<To>;

// Identity when the real part is not needed.
template<typename From, bool>
struct maybe_real {
    static From apply(From src) {
        return src;
    }
};

// Extracts the real part for complex->non-complex casts.
template<typename From>
struct maybe_real<From, true> {
    static decltype(auto) apply(From src) {
        return src.real();
    }
};

// Identity when the boolean reduction is not needed.
template<typename From, bool>
struct maybe_bool {
    static From apply(From src) {
        return src;
    }
};

// Reduces a complex value to (real || imag) for complex->bool casts.
template<typename From>
struct maybe_bool<From, true> {
    static decltype(auto) apply(From src) {
        return src.real() || src.imag();
    }
};

/// Casts a value from From to To.
///
/// For complex->non-complex casts, only the real part is used.
template<typename From, typename To>
struct cast {
    static To apply(From src) {
        constexpr bool real = only_need_real<From, To>;
        auto r = maybe_real<From, real>::apply(src);
        return static_cast<To>(r);
    }
};

template<typename From>
struct cast<From, bool> {
    static bool apply(From src) {
        constexpr bool complex = only_need_real<From, bool>;
        auto r = maybe_bool<From, complex>::apply(src);
        return static_cast<bool>(r);
    }
};

// Route through int64_t to avoid undefined wraparound on negative inputs.
template<typename From>
struct cast<From, uint8_t> {
    static uint8_t apply(From src) {
        constexpr bool real = only_need_real<From, uint8_t>;
        auto r = maybe_real<From, real>::apply(src);
        return static_cast<uint8_t>(static_cast<int64_t>(r));
    }
};

template<>
struct cast<BFloat16, complex<Half>> {
    static complex<Half> apply(BFloat16 src) {
        return complex<float>{src};
    }
};

template<>
struct cast<Float8_e5m2, complex<Half>> {
    static complex<Half> apply(Float8_e5m2 src) {
        return complex<float>{src};
    }
};

template<>
struct cast<Float8_e4m3fn, complex<Half>> {
    static complex<Half> apply(Float8_e4m3fn src) {
        return complex<float>{src};
    }
};

template<>
struct cast<Half, complex<Half>> {
    static complex<Half> apply(Half src) {
        return complex<float>{src};
    }
};

template<>
struct cast<complex<double>, complex<Half>> {
    static complex<Half> apply(complex<double> src) {
        return static_cast<complex<float>>(src);
    }
};

/// Casts src to To, throwing on overflow.
///
/// @tparam From source type
/// @tparam To target type
/// @param src value to cast
/// @param name type name used in the overflow error message
/// @return the cast value
/// @throws RuntimeError if the conversion would overflow
template<typename From, typename To>
To check_and_cast(From src, const char* name) {
    if (!std::is_same_v<To, bool> && is_cast_overflow<From, To>(src)) {
        AM_THROW(RuntimeError) << "Cannot convert the value to type " << name << " without overflow.";
    }
    return cast<From, To>::apply(src);
}

}// namespace aethermind

#endif// AETHERMIND_DTYPES_CAST_H
