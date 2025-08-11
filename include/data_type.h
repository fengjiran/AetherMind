//
// Created by richard on 6/30/25.
//

#ifndef AETHERMIND_DATA_TYPE_H
#define AETHERMIND_DATA_TYPE_H

#include "tensor_utils.h"

#include <cstdint>
#include <glog/logging.h>
#include <string>

namespace aethermind {

enum class DLDataTypeCode : uint8_t {
    kInt = 0,
    kUInt = 1,
    kBool,
    kFloat,
    kBfloat,
    kFloat8_e3m4,
    kFloat8_e4m3,
    kFloat8_e4m3b11fnuz,
    kFloat8_e4m3fn,
    kFloat8_e4m3fnuz,
    kFloat8_e5m2,
    kFloat8_e5m2fnuz,
    kFloat8_e8m0fnu,
    kFloat6_e2m3fn,
    kFloat6_e3m2fn,
    kFloat4_e2m1fn,
    Undefined,
};

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 *   - float8_e4m3: type_code = 8, bits = 8, lanes = 1 (packed in memory)
 *   - float6_e3m2fn: type_code = 16, bits = 6, lanes = 1 (packed in memory)
 *   - float4_e2m1fn: type_code = 17, bits = 4, lanes = 1 (packed in memory)
 *
 *  When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e.,
 *  for a packed data set D ((D >> (i * bits)) && bit_mask) stores the i-th element.
 */
struct DLDataType {
    // Data type code.
    DLDataTypeCode code;

    // Number of bits per data element.
    // For example, 8 for int8_t, 32 for float.
    int8_t bits;

    // Number of lanes per data element, for vector.
    int16_t lanes;
};

inline std::string TypeCode2Str(const DLDataType& dtype) {
    switch (dtype.code) {
        case DLDataTypeCode::kInt: {
            return "int";
        }

        case DLDataTypeCode::kUInt: {
            return "uint";
        }

        case DLDataTypeCode::kFloat: {
            return "float";
        }

        case DLDataTypeCode::kBfloat: {
            return "bfloat";
        }

        case DLDataTypeCode::kFloat8_e3m4: {
            return "float8_e3m4";
        }

        case DLDataTypeCode::kFloat8_e4m3: {
            return "float8_e4m3";
        }

        case DLDataTypeCode::kFloat8_e4m3b11fnuz: {
            return "float8_e4m3b11fnuz";
        }

        case DLDataTypeCode::kFloat8_e4m3fn: {
            return "float8_e4m3fn";
        }

        case DLDataTypeCode::kFloat8_e4m3fnuz: {
            return "float8_e4m3fnuz";
        }

        case DLDataTypeCode::kFloat8_e5m2: {
            return "float8_e5m2";
        }

        case DLDataTypeCode::kFloat8_e5m2fnuz: {
            return "float8_e5m2fnuz";
        }

        case DLDataTypeCode::kFloat8_e8m0fnu: {
            return "float8_e8m0fnu";
        }

        case DLDataTypeCode::kFloat6_e2m3fn: {
            return "float6_e2m3fn";
        }

        case DLDataTypeCode::kFloat6_e3m2fn: {
            return "float6_e3m2fn";
        }

        case DLDataTypeCode::kFloat4_e2m1fn: {
            return "float4_e2m1fn";
        }

        default: {
            AETHERMIND_THROW(runtime_error) << "Unsupported data type";
        }
    }
}

class DataType {
public:
    DataType() : dtype_({DLDataTypeCode::Undefined, -1, -1}) {}

    explicit DataType(DLDataType dtype) : dtype_(dtype) {}

    DataType(DLDataTypeCode code, int bits, int lanes) {
        dtype_.code = code;
        dtype_.bits = static_cast<int8_t>(bits);
        dtype_.lanes = static_cast<int16_t>(lanes);
    }

    NODISCARD DLDataTypeCode code() const {
        return dtype_.code;
    }

    NODISCARD int bits() const {
        CHECK(code() != DLDataTypeCode::Undefined);
        return dtype_.bits;
    }

    NODISCARD int lanes() const {
        CHECK(code() != DLDataTypeCode::Undefined);
        return dtype_.lanes;
    }

    NODISCARD int nbytes() const {
        CHECK(code() != DLDataTypeCode::Undefined);
        return (bits() * lanes() + 7) / 8;
    }

    static DataType Int(int bits, int lanes = 1) {
        return {DLDataTypeCode::kInt, bits, lanes};
    }

    static DataType Float(int bits, int lanes = 1) {
        return {DLDataTypeCode::kFloat, bits, lanes};
    }

    static DataType Float32() {
        return {DLDataTypeCode::kFloat, 32, 1};
    }

    template<typename T>
    static DataType Make();

    template<typename T>
    NODISCARD bool Match() const {
        return *this == Make<T>();
    }

    NODISCARD bool operator==(const DataType& other) const {
        if (this->code() == DLDataTypeCode::Undefined || other.code() == DLDataTypeCode::Undefined) {
            return this->code() == other.code();
        }
        return this->code() == other.code() && this->bits() == other.bits() && this->lanes() == other.lanes();
    }

    NODISCARD bool operator!=(const DataType& other) const {
        return !operator==(other);
    }

private:
    DLDataType dtype_;
};

#define SCALAR_TYPES_NAME(f) \
    f(bool, Bool);           \
    f(uint8_t, Byte);        \
    f(int8_t, Char);         \
    f(uint16_t, UShort);     \
    f(int16_t, Short);       \
    f(uint32_t, UInt);       \
    f(int, Int);             \
    f(uint64_t, ULong);      \
    f(int64_t, Long);        \
    f(float, Float);         \
    f(double, Double);

// TODO: bfloat, float8, float6 and float4 type need to be defined.
#define SCALAR_TYPE_TO_NAME_AND_CPP_TYPE(f)            \
    f(DLDataTypeCode::kInt, 8, 1, Char, int8_t);       \
    f(DLDataTypeCode::kInt, 16, 1, Short, int16_t);    \
    f(DLDataTypeCode::kInt, 32, 1, Int, int32_t);      \
    f(DLDataTypeCode::kInt, 64, 1, Long, int64_t);     \
    f(DLDataTypeCode::kUInt, 8, 1, Byte, uint8_t);     \
    f(DLDataTypeCode::kUInt, 16, 1, UShort, uint16_t); \
    f(DLDataTypeCode::kUInt, 32, 1, UInt, uint32_t);   \
    f(DLDataTypeCode::kUInt, 64, 1, ULong, uint64_t);  \
    f(DLDataTypeCode::kBool, 8, 1, Bool, bool);        \
    f(DLDataTypeCode::kFloat, 32, 1, Float, float);    \
    f(DLDataTypeCode::kFloat, 64, 1, Double, double);  \
    // f(DLDataTypeCode::kBfloat, 16, 1, float);            \
// f(DLDataTypeCode::kFloat8_e3m4, 8, 1, float);        \
// f(DLDataTypeCode::kFloat8_e4m3, 8, 1, float);        \
// f(DLDataTypeCode::kFloat8_e4m3b11fnuz, 8, 1, float); \
// f(DLDataTypeCode::kFloat8_e4m3fn, 8, 1, float);      \
// f(DLDataTypeCode::kFloat8_e4m3fnuz, 8, 1, float);    \
// f(DLDataTypeCode::kFloat8_e5m2, 8, 1, float);        \
// f(DLDataTypeCode::kFloat8_e5m2fnuz, 8, 1, float);    \
// f(DLDataTypeCode::kFloat8_e8m0fnu, 8, 1, float);     \
// f(DLDataTypeCode::kFloat6_e2m3fn, 6, 1, float);      \
// f(DLDataTypeCode::kFloat6_e3m2fn, 6, 1, float);      \
// f(DLDataTypeCode::kFloat4_e2m1fn, 4, 1, float)


}// namespace atp

#endif//AETHERMIND_DATA_TYPE_H
