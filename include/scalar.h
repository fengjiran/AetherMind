//
// Created by richard on 10/4/25.
//

#ifndef AETHERMIND_SCALAR_H
#define AETHERMIND_SCALAR_H

#include "data_type.h"
#include "cast.h"

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
    Scalar(double val);
    Scalar(float val);
    Scalar(Half val);
    Scalar(BFloat16 val);
    Scalar(Float8_e4m3fn val);
    Scalar(Float8_e5m2 val);

    // complex ctors
    Scalar(complex<Half> val);
    Scalar(complex<float> val);
    Scalar(complex<double> val);

    Scalar(const Scalar& other) : v(other.v), dtype_(other.dtype_) {}

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
        return dtype_.is_int() || dtype_.is_uint();
    }

    NODISCARD bool is_signed_integral() const {
        return dtype_.is_int();
    }

    NODISCARD bool is_unsigned_integral() const {
        return dtype_.is_uint();
    }

    NODISCARD bool is_floating_point() const {
        return dtype_.is_float();
    }

    NODISCARD bool is_bool() const {
        return dtype_.is_bool();
    }

    NODISCARD bool is_complex() const {
        return dtype_.is_complex();
    }

    NODISCARD DataType type() const {
        return dtype_;
    }

    void swap(Scalar& other) noexcept {
        std::swap(v, other.v);
        std::swap(dtype_, other.dtype_);
    }

#define ACCESSOR(type, name)                                           \
    type to##name() const {                                            \
        if (is_signed_integral())                                      \
            return static_cast<type>(v.i);                             \
        else if (is_unsigned_integral())                               \
            return static_cast<type>(v.u);                             \
        else if (is_bool())                                            \
            return static_cast<type>(v.d);                             \
        else {                                                         \
            AETHERMIND_THROW(RuntimeError) << "Unsupported data type"; \
            AETHERMIND_UNREACHABLE();                                  \
        }                                                              \
    }

    SCALAR_TYPES_NAME(ACCESSOR);
#undef ACCESSOR

    template<typename T>
    T to() const = delete;

private:
    union val {
        int64_t i;
        uint64_t u;
        double d{};
        complex<double> z;

        val() = default;
    } v;

    // DLDataTypeCode dtype;
    DataType dtype_;
};

#define DEFINE_TO(T, name)           \
    template<>                       \
    inline T Scalar::to<T>() const { \
        return to##name();           \
    }
SCALAR_TYPES_NAME(DEFINE_TO);
#undef DEFINE_TO

std::ostream& operator<<(std::ostream& out, const Scalar& s);
std::string toString(const Scalar& s);

}// namespace aethermind

#endif//AETHERMIND_SCALAR_H
