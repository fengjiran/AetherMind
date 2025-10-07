//
// Created by richard on 10/6/25.
//

#ifndef AETHERMIND_COMPLEX_H
#define AETHERMIND_COMPLEX_H

#include "utils/half.h"

#include <complex>

namespace aethermind {

template<typename T>
class alignas(sizeof(T) * 2) complex {
public:
    using value_type = T;

    constexpr complex() = default;
    constexpr complex(const T& real, const T& imag = T()) : real_(real), imag_(imag) {}// NOLINT

    template<typename U>
    explicit constexpr complex(const std::complex<U>& other) : complex(other.real(), other.imag()) {}

    // ctors for complex<float> and complex<double>
    template<typename U = T,
             typename = std::enable_if_t<std::is_same_v<U, float>>>
    explicit constexpr complex(const complex<double>& other) : real_(other.real_), imag_(other.imag_) {}

    template<typename U = T,
             typename = std::enable_if_t<std::is_same_v<U, double>>>
    constexpr complex(const complex<float>& other) : real_(other.real_), imag_(other.imag_) {}//NOLINT

    NODISCARD constexpr T real() const {
        return real_;
    }

    NODISCARD constexpr T imag() const {
        return imag_;
    }

    constexpr void real(T value) {
        real_ = value;
    }

    constexpr void imag(T value) {
        imag_ = value;
    }

    constexpr complex& operator=(T real) {
        real_ = real;
        imag_ = 0;
        return *this;
    }

    constexpr complex& operator+=(T real) {
        real_ += real;
        return *this;
    }

    constexpr complex& operator-=(T real) {
        real_ -= real;
        return *this;
    }

    constexpr complex& operator*=(T real) {
        real_ *= real;
        imag_ *= real;
        return *this;
    }

    constexpr complex& operator/=(T real) {
        real_ /= real;
        imag_ /= real;
        return *this;
    }

    template<typename U>
    constexpr complex& operator=(const complex<U>& other) {
        real_ = other.real_;
        imag_ = other.imag_;
        return *this;
    }

    template<typename U>
    constexpr complex& operator+=(const complex<U>& other) {
        real_ += other.real_;
        imag_ += other.imag_;
        return *this;
    }

    template<typename U>
    constexpr complex& operator-=(const complex<U>& other) {
        real_ -= other.real_;
        imag_ -= other.imag_;
        return *this;
    }

    template<typename U>
    constexpr complex& operator*=(const complex<U>& other) {
        T a = real_;
        T b = imag_;
        U c = other.real();
        U d = other.imag();
        real_ = a * c - b * d;
        imag_ = a * d + b * c;
        return *this;
    }

    template<typename U>
    constexpr complex& operator/=(const complex<U>& other) __ubsan_ignore_float_divide_by_zero__ {
        T a = real_;
        T b = imag_;
        U c = other.real();
        U d = other.imag();

        auto abs_c = std::abs(c);
        auto abs_d = std::abs(d);
        if (abs_c >= abs_d) {
            if (abs_c == U(0) && abs_d == U(0)) {
                // divide by zeros should yield a complex inf or nan
                real_ = a / abs_c;
                imag_ = b / abs_d;
            } else {
                auto rat = d / c;
                auto scl = U(1.0) / (c + d * rat);
                real_ = (a + b * rat) * scl;
                imag_ = (b - a * rat) * scl;
            }
        } else {
            auto rat = c / d;
            auto scl = U(1.0) / (d + c * rat);
            real_ = (a * rat + b) * scl;
            imag_ = (b * rat - a) * scl;
        }
        return *this;
    }

    template<typename U>
    constexpr complex& operator=(const std::complex<U>& other) {
        real_ = other.real();
        imag_ = other.imag();
        return *this;
    }

    template<typename U>
    explicit constexpr operator std::complex<U>() const {
        return std::complex<U>(std::complex<T>(real(), imag()));
    }

    explicit constexpr operator bool() const {
        return real_ || imag_;
    }

private:
    T real_ = T(0);
    T imag_ = T(0);

    template<typename U>
    friend class complex;
};

template<typename T>
constexpr complex<T> operator+(const complex<T>& val) {
    return val;
}

template<typename T>
constexpr complex<T> operator-(const complex<T>& val) {
    return complex<T>(-val.real(), -val.imag());
}

template<typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
    auto res = lhs;
    return res += rhs;
}

template<typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const T& rhs) {
    auto res = lhs;
    return res += rhs;
}

template<typename T>
constexpr complex<T> operator+(const T& lhs, const complex<T>& rhs) {
    auto res = rhs;
    return res += lhs;
}

template<typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
    auto res = lhs;
    return res -= rhs;
}

template<typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const T& rhs) {
    auto res = lhs;
    return res -= rhs;
}

template<typename T>
constexpr complex<T> operator-(const T& lhs, const complex<T>& rhs) {
    auto res = -rhs;
    return res += lhs;
}

template<typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
    auto res = lhs;
    return res *= rhs;
}

template<typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const T& rhs) {
    auto res = lhs;
    return res *= rhs;
}

template<typename T>
constexpr complex<T> operator*(const T& lhs, const complex<T>& rhs) {
    auto res = rhs;
    return res *= lhs;
}

template<typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
    auto res = lhs;
    return res /= rhs;
}

template<typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const T& rhs) {
    auto res = lhs;
    return res /= rhs;
}

template<typename T>
constexpr complex<T> operator/(const T& lhs, const complex<T>& rhs) {
    complex<T> res(lhs);
    return res /= rhs;
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator+(const complex<Float>& a, const Int& b) {
    return a + static_cast<Float>(b);
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator+(const Int& a, const complex<Float>& b) {
    return static_cast<Float>(a) + b;
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator-(const complex<Float>& a, const Int& b) {
    return a - static_cast<Float>(b);
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator-(const Int& a, const complex<Float>& b) {
    return static_cast<Float>(a) - b;
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator*(const complex<Float>& a, const Int& b) {
    return a * static_cast<Float>(b);
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator*(const Int& a, const complex<Float>& b) {
    return static_cast<Float>(a) * b;
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator/(const complex<Float>& a, const Int& b) {
    return a / static_cast<Float>(b);
}

template<typename Float, typename Int,
         typename = std::enable_if_t<std::is_floating_point_v<Float> && std::is_integral_v<Int>>>
constexpr complex<Float> operator/(const Int& a, const complex<Float>& b) {
    return static_cast<Float>(a) / b;
}

template<typename T>
constexpr bool operator==(const complex<T>& lhs, const complex<T>& rhs) {
    return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}

template<typename T>
constexpr bool operator==(const complex<T>& lhs, const T& rhs) {
    return lhs.real() == rhs && lhs.imag() == T();
}

template<typename T>
constexpr bool operator==(const T& lhs, const complex<T>& rhs) {
    return lhs == rhs.real() && T() == rhs.imag();
}

template<typename T>
constexpr bool operator!=(const complex<T>& lhs, const complex<T>& rhs) {
    return !(lhs == rhs);
}

template<typename T>
constexpr bool operator!=(const complex<T>& lhs, const T& rhs) {
    return !(lhs == rhs);
}

template<typename T>
constexpr bool operator!=(const T& lhs, const complex<T>& rhs) {
    return !(lhs == rhs);
}

template<typename T>
complex<T> polar(const T& r, const T& theta = T()) {
    return complex<T>(r * std::cos(theta), r * std::sin(theta));
}

template<>
class alignas(4) complex<Half> {
public:
    complex() = default;
    complex(const Half& real, const Half& imag) : real_(real), imag_(imag) {}
    complex(const complex<float>& value) : real_(value.real()), imag_(value.imag()) {}//NOLINT

    operator complex<float>() const { //NOLINT
        return {real_, imag_};
    }

    NODISCARD Half real() const {
        return real_;
    }

    NODISCARD Half imag() const {
        return imag_;
    }

    complex& operator+=(const complex& other) {
        real_ += other.real_;
        imag_ += other.imag_;
        return *this;
    }

    complex& operator-=(const complex& other) {
        real_ -= other.real_;
        imag_ -= other.imag_;
        return *this;
    }

    complex& operator*=(const complex& other) {
        auto a = real_;
        auto b = imag_;
        auto c = other.real_;
        auto d = other.imag_;
        real_ = a * c - b * d;
        imag_ = a * d + b * c;
        return *this;
    }

private:
    Half real_;
    Half imag_;
};

}// namespace aethermind

namespace std {
template<typename T>
constexpr T real(const aethermind::complex<T>& x) {
    return x.real();
}

template<typename T>
constexpr T imag(const aethermind::complex<T>& x) {
    return x.imag();
}

template<typename T>
T abs(const aethermind::complex<T>& x) {
    return std::abs(static_cast<std::complex<T>>(x));
}

template<typename T>
constexpr T arg(const aethermind::complex<T>& x) {
    return std::atan2(std::imag(x), std::real(x));
}

template<typename T>
constexpr T norm(const aethermind::complex<T>& x) {
    return x.real() * x.real() + x.imag() * x.imag();
}

template<typename T>
constexpr aethermind::complex<T> conj(const aethermind::complex<T>& x) {
    return aethermind::complex<T>(x.real(), -x.imag());
}

}// namespace std

#endif//AETHERMIND_COMPLEX_H
