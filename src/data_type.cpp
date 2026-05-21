//
// Created by 赵丹 on 25-7-2.
//

#include "data_type.h"

namespace aethermind {

#define DEFINE_MAKE(code, bits, lanes, T, name) \
    template<>                                  \
    DataType DataType::Make<T>() {              \
        return {code, bits, lanes};             \
    }
SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_MAKE);
#undef DEFINE_MAKE

namespace {

void ValidateCodeBitsConsistency(DLDataTypeCode code, int bits) {
    if (code == DLDataTypeCode::Undefined) {
        AM_CHECK(bits == 0, "Undefined type code requires bits==0, got bits={}", bits);
        return;
    }

    AM_CHECK(bits > 0 || code == DLDataTypeCode::kOpaqueHandle,
             "DataType bits must be positive except for void/opaque handles, got bits={}",
             bits);

    if (code == DLDataTypeCode::kBFloat) {
        AM_CHECK(bits == 16, "kBFloat requires bits==16, got bits={}", bits);
        return;
    }

    if (code == DLDataTypeCode::kFloat8_e3m4 || code == DLDataTypeCode::kFloat8_e4m3 ||
        code == DLDataTypeCode::kFloat8_e4m3b11fnuz || code == DLDataTypeCode::kFloat8_e4m3fn ||
        code == DLDataTypeCode::kFloat8_e4m3fnuz || code == DLDataTypeCode::kFloat8_e5m2 ||
        code == DLDataTypeCode::kFloat8_e5m2fnuz || code == DLDataTypeCode::kFloat8_e8m0fnu) {
        AM_CHECK(bits == 8, "Float8 type codes require bits==8, got bits={}", bits);
        return;
    }

    if (code == DLDataTypeCode::kFloat6_e2m3fn || code == DLDataTypeCode::kFloat6_e3m2fn) {
        AM_CHECK(bits == 6, "Float6 type codes require bits==6, got bits={}", bits);
        return;
    }

    if (code == DLDataTypeCode::kFloat4_e2m1fn) {
        AM_CHECK(bits == 4, "kFloat4_e2m1fn requires bits==4, got bits={}", bits);
    }
}

}// namespace

DataType::DataType(DLDataType dtype) : dtype_(dtype) {
    ValidateCodeBitsConsistency(dtype.code, dtype.bits);
}

DataType::DataType(DLDataTypeCode code, int bits, int lanes, bool is_scalable) {
    dtype_.code = code;
    dtype_.bits = static_cast<uint8_t>(bits);

    if (is_scalable) {
        AM_CHECK(lanes > 1, "Invalid value for vscale factor {}", lanes);
    } else {
        AM_CHECK(lanes >= 0, "Fixed-length DataType lanes must be non-negative, got {}", lanes);
    }

    dtype_.lanes = is_scalable ? static_cast<uint16_t>(-lanes) : static_cast<uint16_t>(lanes);

    ValidateCodeBitsConsistency(code, bits);
}

String DataTypeToString(const DataType& dtype) {
    if (dtype.code() == DLDataTypeCode::kUInt && dtype.bits() == 1 && dtype.raw_lanes() == 1) {
        return "bool";
    }

    if (dtype.IsVoid()) {
        return "void";
    }

    std::ostringstream os;
#define GET_NAME(c, b, lanes, T, name)            \
    if (dtype.code() == c && dtype.bits() == b) { \
        os << #name;                              \
    }
    SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(GET_NAME)
#undef GET_NAME

    auto lanes = static_cast<int16_t>(dtype.raw_lanes());
    if (lanes > 1) {
        os << 'x' << lanes;
    } else if (lanes < -1) {
        os << "xvscalex" << -lanes;
    }
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    os << DataTypeToString(dtype);
    return os;
}


}// namespace aethermind
