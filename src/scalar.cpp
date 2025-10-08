//
// Created by richard on 10/4/25.
//

#include "scalar.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace aethermind {

Scalar::Scalar(int8_t val) {
    v.i = static_cast<decltype(v.i)>(val);
    dtype_ = DataType::Int(8);
}

Scalar::Scalar(int16_t val) {
    v.i = static_cast<decltype(v.i)>(val);
    dtype_ = DataType::Int(16);
}

Scalar::Scalar(int32_t val) {
    v.i = static_cast<decltype(v.i)>(val);
    dtype_ = DataType::Int(32);
}

Scalar::Scalar(int64_t val) {
    v.i = val;
    dtype_ = DataType::Int(64);
}

Scalar::Scalar(bool val) {
    v.u = static_cast<decltype(v.u)>(val);
    dtype_ = DataType::Bool();
}

Scalar::Scalar(uint8_t val) {
    v.u = static_cast<decltype(v.u)>(val);
    dtype_ = DataType::UInt(8);
}

Scalar::Scalar(uint16_t val) {
    v.u = static_cast<decltype(v.u)>(val);
    dtype_ = DataType::UInt(16);
}

Scalar::Scalar(uint32_t val) {
    v.u = static_cast<decltype(v.u)>(val);
    dtype_ = DataType::UInt(32);
}

Scalar::Scalar(uint64_t val) {
    v.u = val;
    dtype_ = DataType::UInt(64);
}

Scalar::Scalar(double val) {
    v.d = val;
    dtype_ = DataType::Double();
}

Scalar::Scalar(float val) {
    v.d = static_cast<double>(val);
    dtype_ = DataType::Float32();
}

Scalar::Scalar(Half val) {
    v.d = static_cast<float>(val);
    dtype_ = DataType::Float(16);
}

Scalar::Scalar(BFloat16 val) {
    v.d = static_cast<float>(val);
    dtype_ = DataType::BFloat(16);
}

Scalar::Scalar(Float8_e4m3fn val) {
    v.d = static_cast<float>(val);
    dtype_ = DataType::Float8E4M3FN();
}

Scalar::Scalar(Float8_e5m2 val) {
    v.d = static_cast<float>(val);
    dtype_ = DataType::Float8E5M2();
}

Scalar::Scalar(complex<Half> val) {
    v.z = val;
    dtype_ = DataType::ComplexHalf();
}

Scalar::Scalar(complex<float> val) {
    v.z = val;
    dtype_ = DataType::ComplexFloat();
}

Scalar::Scalar(complex<double> val) {
    v.z = val;
    dtype_ = DataType::ComplexDouble();
}


std::ostream& operator<<(std::ostream& out, const Scalar& s) {
    if (s.is_floating_point()) {
        return out << s.toDouble();
    }

    if (s.is_bool()) {
        return out << (s.toBool() ? "true" : "false");
    }

    if (s.is_integral()) {
        return out << s.toLong();
    }

    AETHERMIND_THROW(RuntimeError) << "Unknown type in Scalar";
    AETHERMIND_UNREACHABLE();
}

std::string toString(const Scalar& s) {
    return fmt::format("{}", fmt::streamed(s));
}
}// namespace aethermind