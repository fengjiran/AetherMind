//
// Created by richard on 10/4/25.
//

#include "scalar.h"

namespace aethermind {

Scalar::Scalar(int8_t val) {
    v.i = cast<int8_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(8);
}

Scalar::Scalar(int16_t val) {
    v.i = cast<int16_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(16);
}

Scalar::Scalar(int32_t val) {
    v.i = cast<int32_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(32);
}

Scalar::Scalar(int64_t val) {
    v.i = cast<int64_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(64);
}

Scalar::Scalar(bool val) {
    v.i = cast<bool, decltype(v.i)>::apply(val);
    dtype_ = DataType::Bool();
}

Scalar::Scalar(uint8_t val) {
    v.i = cast<uint8_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(8);
}

Scalar::Scalar(uint16_t val) {
    v.i = cast<uint16_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(16);
}

Scalar::Scalar(uint32_t val) {
    v.i = cast<uint32_t, decltype(v.i)>::apply(val);
    dtype_ = DataType::Int(32);
}

Scalar::Scalar(uint64_t val) {
    if (val > static_cast<uint64_t>(INT64_MAX)) {
        v.u = val;
        dtype_ = DataType::UInt(64);
    } else {
        v.i = static_cast<int64_t>(val);
        dtype_ = DataType::Int(64);
    }
}

Scalar::Scalar(double val) {
    v.d = cast<double, decltype(v.d)>::apply(val);
    dtype_ = DataType::Double();
}

Scalar::Scalar(float val) {
    v.d = cast<float, decltype(v.d)>::apply(val);
    dtype_ = DataType::Float32();
}

Scalar::Scalar(Half val) {
    v.d = cast<Half, decltype(v.d)>::apply(val);
    dtype_ = DataType::Float(16);
}

Scalar::Scalar(BFloat16 val) {
    v.d = cast<BFloat16, decltype(v.d)>::apply(val);
    dtype_ = DataType::BFloat(16);
}

Scalar::Scalar(Float8_e4m3fn val) {
    v.d = cast<Float8_e4m3fn, decltype(v.d)>::apply(val);
    dtype_ = DataType::Float8E4M3FN();
}

Scalar::Scalar(Float8_e5m2 val) {
    v.d = cast<Float8_e5m2, decltype(v.d)>::apply(val);
    dtype_ = DataType::Float8E5M2();
}

Scalar::Scalar(complex<Half> val) {
    v.z = cast<complex<Half>, decltype(v.z)>::apply(val);
    dtype_ = DataType::ComplexHalf();
}

Scalar::Scalar(complex<float> val) {
    v.z = cast<complex<float>, decltype(v.z)>::apply(val);
    dtype_ = DataType::ComplexFloat();
}

Scalar::Scalar(complex<double> val) {
    v.z = cast<complex<double>, decltype(v.z)>::apply(val);
    dtype_ = DataType::ComplexDouble();
}

Scalar Scalar::operator-() const {
    AM_CHECK(!is_bool(), "boolean negative is not supported.");
    if (is_signed_integral()) {
        return {-v.i};
    }

    if (is_floating_point()) {
        return {-v.d};
    }

    if (is_complex()) {
        return {-v.z};
    }

    AM_THROW(RuntimeError) << dtype_ << " is not supported.";
    AM_UNREACHABLE();
}

Scalar Scalar::log() const {
    if (is_integral()) {
        return std::log(v.i);
    }

    if (is_floating_point()) {
        return std::log(v.d);
    }

    if (is_complex()) {
        return std::log(v.z);
    }

    AM_THROW(RuntimeError) << dtype_ << " is not supported.";
    AM_UNREACHABLE();
}

Scalar Scalar::conj() const {
    if (is_complex()) {
        return {std::conj(v.z)};
    }
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Scalar& s) {
    if (s.is_floating_point()) {
        os << s.toDouble();
    }

    if (s.is_bool()) {
        os << (s.toBool() ? "true" : "false");
    }

    if (s.is_signed_integral()) {
        os << s.toLong();
    }

    if (s.is_unsigned_integral()) {
        os << s.toUInt64();
    }

    if (s.is_complex()) {
        os << s.toComplexDouble();
    }

    return os;

    AM_THROW(RuntimeError) << "Unknown type in Scalar";
    AM_UNREACHABLE();
}

// String toString(const Scalar& s) {
//     return std::format("{}", s);
// }
}// namespace aethermind