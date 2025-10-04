//
// Created by 赵丹 on 25-7-2.
//

#include "data_type.h"

namespace aethermind {

#define DEFINE_MAKE(code, bits, lanes, T, name) \
    template<>                                  \
    DataType DataType::Make<T>() {              \
        return DataType(code, bits, lanes);     \
    }
SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_MAKE);
#undef DEFINE_MAKE

DataType::DataType(DLDataTypeCode code, int bits, int lanes, bool is_scalable) {
    dtype_.code = code;
    dtype_.bits = static_cast<uint8_t>(bits);

    if (is_scalable) {
        CHECK(lanes > 1) << "Invalid value for vscale factor" << lanes;
    }

    dtype_.lanes = is_scalable ? static_cast<uint16_t>(-lanes) : static_cast<uint16_t>(lanes);

    if (code == DLDataTypeCode::kBFloat) {
        CHECK(bits == 16);
    }

    if (code == DLDataTypeCode::kFloat8_e3m4 || code == DLDataTypeCode::kFloat8_e4m3 ||
        code == DLDataTypeCode::kFloat8_e4m3b11fnuz || code == DLDataTypeCode::kFloat8_e4m3fn ||
        code == DLDataTypeCode::kFloat8_e4m3fnuz || code == DLDataTypeCode::kFloat8_e5m2 ||
        code == DLDataTypeCode::kFloat8_e5m2fnuz || code == DLDataTypeCode::kFloat8_e8m0fnu) {
        CHECK(bits == 8);
    }

    if (code == DLDataTypeCode::kFloat6_e2m3fn || code == DLDataTypeCode::kFloat6_e3m2fn) {
        CHECK(bits == 6);
    }

    if (code == DLDataTypeCode::kFloat4_e2m1fn) {
        CHECK(bits == 4);
    }
}


}// namespace aethermind