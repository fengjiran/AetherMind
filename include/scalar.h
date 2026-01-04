//
// Created by richard on 10/4/25.
//

#ifndef AETHERMIND_SCALAR_H
#define AETHERMIND_SCALAR_H

#include "cast.h"
#include "data_type.h"

namespace aethermind {

/// Scalar represents a 0-dimensional tensor which contains a single element.
/// Unlike a tensor, numeric literals (in C++) are implicitly convertible to
/// Scalar (which is why, for example, we provide both add(Tensor) and
/// add(Scalar) overloads for many operations). It may also be used in
/// circumstances where you statically know a tensor is 0-dim and single size,
/// but don't know its type.
class Scalar {
public:
    Scalar() : Scalar(static_cast<int64_t>(0)) {}

    // integral ctors
    Scalar(int8_t val);  //NOLINT
    Scalar(int16_t val); //NOLINT
    Scalar(int32_t val); //NOLINT
    Scalar(int64_t val); //NOLINT
    Scalar(bool val);    //NOLINT
    Scalar(uint8_t val); //NOLINT
    Scalar(uint16_t val);//NOLINT
    Scalar(uint32_t val);//NOLINT
    Scalar(uint64_t val);//NOLINT

    // floating point ctors
    Scalar(double val);       //NOLINT
    Scalar(float val);        //NOLINT
    Scalar(Half val);         //NOLINT
    Scalar(BFloat16 val);     //NOLINT
    Scalar(Float8_e4m3fn val);//NOLINT
    Scalar(Float8_e5m2 val);  //NOLINT

    // complex ctors
    Scalar(complex<Half> val);  //NOLINT
    Scalar(complex<float> val); //NOLINT
    Scalar(complex<double> val);//NOLINT

    Scalar(const Scalar& other) = default;

    Scalar(Scalar&& other) noexcept : v(other.v), dtype_(other.dtype_) {
        other.v.i = 0;
        other.dtype_ = {};
    }

    Scalar& operator=(const Scalar& other) {
        Scalar(other).swap(*this);
        return *this;
    }

    Scalar& operator=(Scalar&& other) noexcept {
        Scalar(std::move(other)).swap(*this);
        return *this;
    }

    NODISCARD bool is_integral() const {
        return dtype_.IsInt() || dtype_.IsUint();
    }

    NODISCARD bool is_signed_integral() const {
        return dtype_.IsInt();
    }

    NODISCARD bool is_unsigned_integral() const {
        return dtype_.IsUint();
    }

    NODISCARD bool is_floating_point() const {
        return dtype_.IsFloat();
    }

    NODISCARD bool is_bool() const {
        return dtype_.IsBool();
    }

    NODISCARD bool is_complex() const {
        return dtype_.IsComplex();
    }

    NODISCARD DataType type() const {
        return dtype_;
    }

    void swap(Scalar& other) noexcept {
        std::swap(v, other.v);
        std::swap(dtype_, other.dtype_);
    }

    Scalar operator-() const;
    NODISCARD Scalar log() const;
    NODISCARD Scalar conj() const;

#define ACCESSOR(code, bits, lanes, type, name)                        \
    type to##name() const {                                            \
        if (is_signed_integral())                                      \
            return check_and_cast<int64_t, type>(v.i, #type);          \
        else if (is_unsigned_integral())                               \
            return check_and_cast<uint64_t, type>(v.u, #type);         \
        else if (is_bool())                                            \
            return check_and_cast<bool, type>(v.u, #type);             \
        else if (is_floating_point())                                  \
            return check_and_cast<double, type>(v.d, #type);           \
        else if (is_complex())                                         \
            return check_and_cast<complex<double>, type>(v.z, #type);  \
        else {                                                         \
            AETHERMIND_THROW(RuntimeError) << "Unsupported data type"; \
            AETHERMIND_UNREACHABLE();                                  \
        }                                                              \
    }

    SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(ACCESSOR);
#undef ACCESSOR

    template<typename T>
    T to() const = delete;

    template<typename T, std::enable_if_t<!is_complex_v<T>>* = nullptr>
    bool equal(T x) const {
        if (is_complex()) {
            const auto val = v.z;
            return val.real() == x && val.imag() == T();
        }

        if (is_floating_point()) {
            return toDouble() == x;
        }

        if (is_signed_integral()) {
            if (is_overflow<decltype(v.i), T>(v.i, true)) {
                return false;
            }
            return static_cast<T>(v.i) == x;
        }

        if (is_unsigned_integral()) {
            if (is_overflow<decltype(v.u), T>(v.u, true)) {
                return false;
            }
            return static_cast<T>(v.u) == x;
        }

        if (is_bool()) {
            return false;
        }

        AETHERMIND_THROW(RuntimeError) << "Unsupported data type";
        AETHERMIND_UNREACHABLE();
    }

    template<typename T, std::enable_if_t<is_complex_v<T>>* = nullptr>
    bool equal(T x) const {
        if (is_complex()) {
            return x == v.z;
        }

        if (is_floating_point()) {
            return toDouble() == x.real() && x.imag() == T();
        }

        if (is_signed_integral()) {
            if (is_overflow<decltype(v.i), T>(v.i, true)) {
                return false;
            }

            return static_cast<T>(v.i) == x.real() && x.imag() == T();
        }

        if (is_unsigned_integral()) {
            if (is_overflow<decltype(v.u), T>(v.u, true)) {
                return false;
            }

            return static_cast<T>(v.u) == x.real() && x.imag() == T();
        }

        if (is_bool()) {
            return false;
        }

        AETHERMIND_THROW(RuntimeError) << "Unsupported data type";
        AETHERMIND_UNREACHABLE();
    }

    NODISCARD bool equal(bool x) const {
        if (is_bool()) {
            return static_cast<bool>(v.i) == x;
        }
        return false;
    }

private:
    union val {
        int64_t i;
        uint64_t u;
        double d{};
        complex<double> z;

        val() = default;
    } v;

    DataType dtype_;
};

#define DEFINE_TO(code, bits, lanes, T, name) \
    template<>                                \
    inline T Scalar::to<T>() const {          \
        return to##name();                    \
    }

SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_TO);
#undef DEFINE_TO

std::ostream& operator<<(std::ostream& os, const Scalar& s);
// String toString(const Scalar& s);

template<typename T>
bool operator==(const Scalar& lhs, T rhs) {
    return lhs.equal(rhs);
}

template<typename T>
bool operator==(T lhs, const Scalar& rhs) {
    return rhs.equal(lhs);
}

}// namespace aethermind

#endif//AETHERMIND_SCALAR_H
