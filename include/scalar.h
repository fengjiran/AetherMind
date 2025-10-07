//
// Created by richard on 10/4/25.
//

#ifndef AETHERMIND_SCALAR_H
#define AETHERMIND_SCALAR_H

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

    template<typename T,
             std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>* = nullptr>
    Scalar(T val) {//NOLINT
        v.i = static_cast<decltype(v.i)>(val);
        dtype = DLDataTypeCode::kInt;
    }

    template<typename T,
             std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    Scalar(T val) {//NOLINT
        v.d = static_cast<decltype(v.d)>(val);
        dtype = DLDataTypeCode::kFloat;
    }

    template<typename T,
             std::enable_if_t<std::is_same_v<T, bool>>* = nullptr>
    Scalar(T val) {//NOLINT
        v.i = static_cast<decltype(v.i)>(val);
        dtype = DLDataTypeCode::kBool;
    }

    Scalar(const Scalar& other) : v(other.v), dtype(other.dtype) {}//NOLINT

    Scalar(Scalar&& other) noexcept : v(other.v), dtype(other.dtype) {
        other.v.i = 0;
        other.dtype = DLDataTypeCode::kInt;
    }

    Scalar& operator=(const Scalar& other) {
        Scalar tmp(other);
        swap(*this, tmp);
        return *this;
    }

    Scalar& operator=(Scalar&& other) noexcept {
        Scalar tmp(std::move(other));
        swap(*this, tmp);
        return *this;
    }

    NODISCARD bool is_integral() const {
        return dtype == DLDataTypeCode::kInt;
    }

    NODISCARD bool is_floating_point() const {
        return dtype == DLDataTypeCode::kFloat;
    }

    NODISCARD bool is_bool() const {
        return dtype == DLDataTypeCode::kBool;
    }

    NODISCARD DLDataTypeCode type() const {
        return dtype;
    }

    friend void swap(Scalar& a, Scalar& b) noexcept {
        std::swap(a.v, b.v);
        std::swap(a.dtype, b.dtype);
    }

#define ACCESSOR(type, name)                                                 \
    type to##name() const {                                                  \
        if (dtype == DLDataTypeCode::kInt || dtype == DLDataTypeCode::kBool) \
            return static_cast<type>(v.i);                                   \
        else if (dtype == DLDataTypeCode::kFloat)                            \
            return static_cast<type>(v.d);                                   \
        else {                                                               \
            AETHERMIND_THROW(RuntimeError) << "Unsupported data type";       \
            AETHERMIND_UNREACHABLE();                                        \
        }                                                                    \
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

    DLDataTypeCode dtype;
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
