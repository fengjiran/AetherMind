//
// Created by richard on 10/6/25.
//

#ifndef AETHERMIND_COMPLEX_H
#define AETHERMIND_COMPLEX_H

#include "utils/half.h"

#include <complex>

namespace aethermind {

template<typename T>
struct alignas(sizeof(T) * 2) complex {
    using value_type = T;

    T real_ = T(0);
    T imag_ = T(0);

    constexpr complex() = default;
    constexpr complex(const T& real, const T& imag = T()) : real_(real), imag_(imag) {}// NOLINT

    template<typename U>
    explicit constexpr complex(const complex<U>& other) : complex(other.real_, other.imag_) {}

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
};

}// namespace aethermind


#endif//AETHERMIND_COMPLEX_H
