//
// Created by richard on 10/8/25.
//

#ifndef AETHERMIND_CAST_H
#define AETHERMIND_CAST_H

#include "error.h"
#include "macros.h"
#include "utils/bfloat16.h"
#include "utils/complex.h"
#include "utils/float8_e4m3fn.h"
#include "utils/float8_e5m2.h"

namespace aethermind {

// Returns false since we cannot have x < 0 if x is unsigned.
template<typename T>
constexpr bool is_negative(const T&, std::true_type) {
    return false;
}

// Returns true if a signed variable x < 0
template<typename T>
constexpr bool is_negative(const T& x, std::false_type) {
    return x < T(0);
}

/// Returns true if x < 0
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, Half does not :-(
template<typename T>
constexpr bool is_negative(const T& x) {
    return is_negative(x, std::is_unsigned<T>());
}

// Returns true if a and b are not both negative
template<typename T, typename U>
constexpr bool signs_differ(const T& a, const U& b) {
    return is_negative(a) != is_negative(b);
}

// Returns the sign of an unsigned variable x as 0, 1
template<typename T>
constexpr int signum(const T& x, std::true_type) {
    return T(0) < x;
}

// Returns the sign of a signed variable x as -1, 0, 1
template<typename T>
constexpr int signum(const T& x, std::false_type) {
    return (T(0) < x) - (x < T(0));
}

/// Returns the sign of x as -1, 0, 1
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, Half does not :-(
template<typename T>
constexpr int signum(const T& x) {
    return signum(x, std::is_unsigned<T>());
}

// Returns true if x is greater than the greatest value of the type Limit
template<typename Limit, typename T>
constexpr bool greater_than_max(const T& x) {
    constexpr bool can_overflow = std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
    return can_overflow && x > std::numeric_limits<Limit>::max();
}

// Returns true if x < lowest(Limit). Standard comparison
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& x,
        std::false_type /*limit_is_unsigned*/,
        std::false_type /*x_is_unsigned*/) {
    return x < std::numeric_limits<Limit>::lowest();
}

/// Returns false since all the limit is signed and therefore includes
/// negative values but x cannot be negative because it is unsigned
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& /*x*/,
        std::false_type /*limit_is_unsigned*/,
        std::true_type /*x_is_unsigned*/) {
    return false;
}

/// Returns true if x < 0, where 0 is constructed from T.
/// Limit is not signed, so its lower value is zero
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& x,
        std::true_type /*limit_is_unsigned*/,
        std::false_type /*x_is_unsigned*/) {
    return x < T(0);
}

/// Returns false sign both types are unsigned
template<typename Limit, typename T>
constexpr bool less_than_lowest(
        const T& /*x*/,
        std::true_type /*limit_is_unsigned*/,
        std::true_type /*x_is_unsigned*/) {
    return false;
}

/// Returns true if x is less than the lowest value of type T
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :
template<typename Limit, typename T>
constexpr bool less_than_lowest(const T& x) {
    return less_than_lowest<Limit>(x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

// bool can be converted to any type
template<typename From, typename To,
         std::enable_if_t<std::is_same_v<From, bool>>* = nullptr>
bool is_overflow(From, MAYBE_UNUSED bool strict_unsigned = false) {
    return false;
}

template<typename From, typename To,
         std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>>* = nullptr>
bool is_overflow(From src, bool strict_unsigned = false) {
    using Limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if constexpr (!Limit::is_signed && std::numeric_limits<From>::is_signed) {
        if (!strict_unsigned) {
            return greater_than_max<To>(src) ||
                   (is_negative(src) && -static_cast<uint64_t>(src) > static_cast<uint64_t>(Limit::max()));
        }
    }

    return greater_than_max<To>(src) || less_than_lowest<To>(src);
}

template<typename From, typename To,
         std::enable_if_t<std::is_floating_point_v<From>>* = nullptr>
bool is_overflow(From src, MAYBE_UNUSED bool strict_unsigned = false) {
    using Limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if (Limit::has_infinity && std::isinf(static_cast<double>(src))) {
        return false;
    }

    if (!Limit::has_quiet_NaN && src != src) {
        return true;
    }
    return src < Limit::lowest() || src > Limit::max();
}

template<typename From, typename To, std::enable_if_t<is_complex_v<From>>* = nullptr>
bool is_overflow(From src, bool strict_unsigned = false) {
    if (!is_complex_v<To> && src.imag() != 0) {
        return true;
    }

    using from_type = From::value_type;
    using to_type = scalar_value_type<To>::type;

    return is_overflow<from_type, to_type>(src.real(), strict_unsigned) ||
           is_overflow<from_type, to_type>(src.imag(), strict_unsigned);
}

template<typename From, typename To>
constexpr static bool only_need_real = is_complex_v<From> && !is_complex_v<To>;

template<typename From, bool>
struct maybe_real {
    static From apply(From src) {
        return src;
    }
};

template<typename From>
struct maybe_real<From, true> {
    static decltype(auto) apply(From src) {
        return src.real();
    }
};

template<typename From, bool>
struct maybe_bool {
    static From apply(From src) {
        return src;
    }
};

template<typename From>
struct maybe_bool<From, true> {
    static decltype(auto) apply(From src) {
        return src.real() || src.imag();
    }
};

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

template<typename From, typename To>
To check_and_cast(From src, const char* name) {
    if (!std::is_same_v<To, bool> && is_overflow<From, To>(src)) {
        AETHERMIND_THROW(RuntimeError) << "Cannot convert the value to type " << name << " without overflow.";
    }
    return cast<From, To>::apply(src);
}


}// namespace aethermind

#endif//AETHERMIND_CAST_H
