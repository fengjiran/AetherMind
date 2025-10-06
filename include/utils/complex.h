//
// Created by richard on 10/6/25.
//

#ifndef AETHERMIND_COMPLEX_H
#define AETHERMIND_COMPLEX_H

#include "utils/half.h"

namespace aethermind {

template<typename T>
struct alignas(sizeof(T) * 2) complex {
    using value_type = T;

    T real_ = T(0);
    T imag_ = T(0);

    constexpr complex() = default;
    constexpr complex(const T& real, const T& imag = T()) : real_(real), imag_(imag) {}// NOLINT

    template<typename U>
    explicit constexpr complex(const complex<U>& other) : real_(other.real()), imag_(other.imag()) {}


    constexpr T real() const {
        return real_;
    }

    constexpr T imag() const {
        return imag_;
    }
};

}// namespace aethermind


#endif//AETHERMIND_COMPLEX_H
