//
// Created by richard on 6/30/25.
//

#ifndef AETHERMIND_DATA_TYPE_H
#define AETHERMIND_DATA_TYPE_H

#include "error.h"
#include "macros.h"
#include "utils/bfloat16.h"
#include "utils/bits.h"
#include "utils/complex.h"
#include "utils/float8_e4m3fn.h"
#include "utils/float8_e5m2.h"
#include "utils/half.h"

#include <cstdint>
#include <glog/logging.h>
#include <string>

namespace aethermind {

enum class DLDataTypeCode : uint8_t {
    kInt = 0,
    kUInt = 1,
    kBool,
    kOpaqueHandle,
    kFloat,
    kBFloat,
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
    kComplex,
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
    uint8_t bits;

    // Number of lanes per data element, for vector.
    uint16_t lanes;
};

// inline std::string TypeCode2Str(const DLDataType& dtype) {
//     switch (dtype.code) {
//         case DLDataTypeCode::kInt: {
//             return "int";
//         }
//
//         case DLDataTypeCode::kUInt: {
//             return "uint";
//         }
//
//         case DLDataTypeCode::kFloat: {
//             return "float";
//         }
//
//         case DLDataTypeCode::kHalf: {
//             return "half";
//         }
//
//         case DLDataTypeCode::kBfloat: {
//             return "bfloat16";
//         }
//
//         case DLDataTypeCode::kFloat8_e3m4: {
//             return "float8_e3m4";
//         }
//
//         case DLDataTypeCode::kFloat8_e4m3: {
//             return "float8_e4m3";
//         }
//
//         case DLDataTypeCode::kFloat8_e4m3b11fnuz: {
//             return "float8_e4m3b11fnuz";
//         }
//
//         case DLDataTypeCode::kFloat8_e4m3fn: {
//             return "float8_e4m3fn";
//         }
//
//         case DLDataTypeCode::kFloat8_e4m3fnuz: {
//             return "float8_e4m3fnuz";
//         }
//
//         case DLDataTypeCode::kFloat8_e5m2: {
//             return "float8_e5m2";
//         }
//
//         case DLDataTypeCode::kFloat8_e5m2fnuz: {
//             return "float8_e5m2fnuz";
//         }
//
//         case DLDataTypeCode::kFloat8_e8m0fnu: {
//             return "float8_e8m0fnu";
//         }
//
//         case DLDataTypeCode::kFloat6_e2m3fn: {
//             return "float6_e2m3fn";
//         }
//
//         case DLDataTypeCode::kFloat6_e3m2fn: {
//             return "float6_e3m2fn";
//         }
//
//         case DLDataTypeCode::kFloat4_e2m1fn: {
//             return "float4_e2m1fn";
//         }
//
//         default: {
//             AETHERMIND_THROW(runtime_error) << "Unsupported data type";
//             AETHERMIND_UNREACHABLE();
//         }
//     }
// }

class DataType {
public:
    DataType() : dtype_({DLDataTypeCode::Undefined, 0, 0}) {}

    explicit DataType(DLDataType dtype) : dtype_(dtype) {}

    DataType(DLDataTypeCode code, int bits, int lanes, bool is_scalable = false);

    DataType& operator=(const DataType& other) {
        if (this != &other) {
            dtype_ = other.dtype_;
        }
        return *this;
    }

    operator DLDataType() const {// NOLINT
        return dtype_;
    }

    NODISCARD DLDataTypeCode code() const {
        return dtype_.code;
    }

    NODISCARD int bits() const {
        return dtype_.bits;
    }

    NODISCARD int lanes() const {
        int lanes_as_int = static_cast<int16_t>(dtype_.lanes);
        if (lanes_as_int < 0) {
            LOG(FATAL) << "Can't fetch the lanes of a scalable vector at compile time.";
        }
        return lanes_as_int;
    }

    NODISCARD uint16_t raw_lanes() const {
        return dtype_.lanes;
    }

    NODISCARD int vscale_factor() const {
        int lanes_as_int = static_cast<int16_t>(dtype_.lanes);
        if (lanes_as_int >= -1) {
            LOG(FATAL) << "A fixed length vector doesn't have a vscale factor.";
        }
        return -lanes_as_int;
    }

    NODISCARD int get_lanes_or_vscale_factor() const {
        return is_scalable_vector() ? vscale_factor() : lanes();
    }

    NODISCARD bool is_scalar() const {
        return !is_scalable_vector() && lanes() == 1;
    }

    NODISCARD bool is_bool() const {
        return code() == DLDataTypeCode::kUInt && bits() == 1;
    }

    NODISCARD bool is_float() const {
        return code() == DLDataTypeCode::kFloat;
    }

    NODISCARD bool is_float8() const {
        return bits() == 8 &&
               (code() == DLDataTypeCode::kFloat8_e3m4 || code() == DLDataTypeCode::kFloat8_e4m3 ||
                code() == DLDataTypeCode::kFloat8_e4m3b11fnuz || code() == DLDataTypeCode::kFloat8_e4m3fn ||
                code() == DLDataTypeCode::kFloat8_e4m3fnuz || code() == DLDataTypeCode::kFloat8_e5m2 ||
                code() == DLDataTypeCode::kFloat8_e5m2fnuz || code() == DLDataTypeCode::kFloat8_e8m0fnu);
    }

    NODISCARD bool is_float6() const {
        return bits() == 6 &&
               (code() == DLDataTypeCode::kFloat6_e2m3fn || code() == DLDataTypeCode::kFloat6_e3m2fn);
    }

    NODISCARD bool is_float4() const {
        return bits() == 4 && code() == DLDataTypeCode::kFloat4_e2m1fn;
    }

    NODISCARD bool is_float8_e3m4() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e3m4;
    }

    NODISCARD bool is_float8_e4m3() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e4m3;
    }

    NODISCARD bool is_float8_e4m3b11fnuz() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e4m3b11fnuz;
    }

    NODISCARD bool is_float8_e4m3fn() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e4m3fn;
    }

    NODISCARD bool is_float8_e4m3fnuz() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e4m3fnuz;
    }

    NODISCARD bool is_float8_e5m2() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e5m2;
    }

    NODISCARD bool is_float8_e5m2fnuz() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e5m2fnuz;
    }

    NODISCARD bool is_float8_e8m0fnu() const {
        return bits() == 8 && code() == DLDataTypeCode::kFloat8_e8m0fnu;
    }

    NODISCARD bool is_float6_e2m3fn() const {
        return bits() == 6 && code() == DLDataTypeCode::kFloat6_e2m3fn;
    }

    NODISCARD bool is_float6_e3m2fn() const {
        return bits() == 6 && code() == DLDataTypeCode::kFloat6_e3m2fn;
    }

    NODISCARD bool is_float4_e2m1fn() const {
        return bits() == 4 && code() == DLDataTypeCode::kFloat4_e2m1fn;
    }

    NODISCARD bool is_float16() const {
        return is_float() && bits() == 16;
    }

    NODISCARD bool is_half() const {
        return is_float16();
    }

    NODISCARD bool is_bfloat16() const {
        return code() == DLDataTypeCode::kBFloat && bits() == 16;
    }

    NODISCARD bool is_int() const {
        return code() == DLDataTypeCode::kInt;
    }

    NODISCARD bool is_uint() const {
        return code() == DLDataTypeCode::kUInt;
    }

    NODISCARD bool is_handle() const {
        return code() == DLDataTypeCode::kOpaqueHandle && !is_void();
    }

    NODISCARD bool is_void() const {
        return code() == DLDataTypeCode::kOpaqueHandle && bits() == 0 && lanes() == 0;
    }

    NODISCARD bool is_vector() const {
        return lanes() > 1;
    }

    NODISCARD bool is_fixed_length_vector() const {
        return static_cast<int16_t>(dtype_.lanes) > 1;
    }

    NODISCARD bool is_scalable_vector() const {
        return static_cast<int16_t>(dtype_.lanes) < -1;
    }

    NODISCARD bool is_scalable_or_fixed_length_vector() const {
        int encoded_lanes = static_cast<int16_t>(dtype_.lanes);
        return encoded_lanes < -1 || encoded_lanes > 1;
    }

    NODISCARD bool is_vector_bool() const {
        return is_scalable_or_fixed_length_vector() && bits() == 1;
    }

    NODISCARD int nbytes() const {
        return (bits() + 7) / 8;
    }

    NODISCARD DataType with_lanes(int lanes) const {
        return {code(), bits(), lanes};
    }

    NODISCARD DataType with_bits(int bits) const {
        return {code(), bits, dtype_.lanes};
    }

    NODISCARD DataType with_scalable_vscale_factor(int vscale_factor) const {
        return {code(), bits(), -vscale_factor};
    }

    NODISCARD DataType element_of() const {
        return with_lanes(1);
    }

    static DataType Int(int bits, int lanes = 1) {
        return {DLDataTypeCode::kInt, bits, lanes};
    }

    static DataType UInt(int bits, int lanes = 1, bool is_scalable = false) {
        return {DLDataTypeCode::kUInt, bits, lanes, is_scalable};
    }

    static DataType Float(int bits, int lanes = 1) {
        return {DLDataTypeCode::kFloat, bits, lanes};
    }

    static DataType Float32() {
        return {DLDataTypeCode::kFloat, 32, 1};
    }

    static DataType BFloat(int bits, int lanes = 1) {
        return {DLDataTypeCode::kBFloat, bits, lanes};
    }

    static DataType Float8E3M4(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e3m4, 8, lanes};
    }

    static DataType Float8E4M3(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e4m3, 8, lanes};
    }

    static DataType Float8E4M3B11FNUZ(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e4m3b11fnuz, 8, lanes};
    }

    static DataType Float8E4M3FN(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e4m3fn, 8, lanes};
    }

    static DataType Float8E4M3FNUZ(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e4m3fnuz, 8, lanes};
    }

    static DataType Float8E5M2(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e5m2, 8, lanes};
    }

    static DataType Float8E5M2FNUZ(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e5m2fnuz, 8, lanes};
    }

    static DataType Float8E8M0FNU(int lanes = 1) {
        return {DLDataTypeCode::kFloat8_e8m0fnu, 8, lanes};
    }

    static DataType Float6E3M2FN(int lanes = 1) {
        return {DLDataTypeCode::kFloat6_e3m2fn, 6, lanes};
    }

    static DataType Float6E2M3FN(int lanes = 1) {
        return {DLDataTypeCode::kFloat6_e2m3fn, 6, lanes};
    }

    static DataType Float4E2M1FN(int lanes = 1) {
        return {DLDataTypeCode::kFloat4_e2m1fn, 4, lanes};
    }

    static DataType Bool(int lanes = 1, bool is_scalable = false) {
        return UInt(1, lanes, is_scalable);
        // return {DLDataTypeCode::kBool, 1, lanes, is_scalable};
    }

    static DataType Handle(int bits = 64, int lanes = 1) {
        return {DLDataTypeCode::kOpaqueHandle, bits, lanes};
    }

    static DataType Void() {
        return {DLDataTypeCode::kOpaqueHandle, 0, 0};
    }

    template<typename T>
    static DataType Make();

    template<typename T>
    NODISCARD bool Match() const {
        return *this == Make<T>();
    }

    NODISCARD bool operator==(const DataType& other) const {
        if (code() == DLDataTypeCode::Undefined || other.code() == DLDataTypeCode::Undefined) {
            return code() == other.code();
        }
        return code() == other.code() && bits() == other.bits() && lanes() == other.lanes();
    }

    NODISCARD bool operator!=(const DataType& other) const {
        return !operator==(other);
    }

private:
    DLDataType dtype_;
};

std::string DataTypeToString(const DataType& dtype);

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

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


// TODO: float8, float6 and float4 type need to be defined.
#define SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(f)                                \
    f(DLDataTypeCode::kInt, 8, 1, int8_t, Char);                           \
    f(DLDataTypeCode::kInt, 16, 1, int16_t, Short);                        \
    f(DLDataTypeCode::kInt, 32, 1, int32_t, Int);                          \
    f(DLDataTypeCode::kInt, 64, 1, int64_t, Long);                         \
    f(DLDataTypeCode::kUInt, 8, 1, uint8_t, Byte);                         \
    f(DLDataTypeCode::kUInt, 16, 1, uint16_t, UShort);                     \
    f(DLDataTypeCode::kUInt, 32, 1, uint32_t, UInt);                       \
    f(DLDataTypeCode::kUInt, 64, 1, uint64_t, ULong);                      \
    f(DLDataTypeCode::kUInt, 1, 1, bool, Bool);                            \
    f(DLDataTypeCode::kFloat, 16, 1, Half, Half);                          \
    f(DLDataTypeCode::kFloat, 32, 1, float, Float);                        \
    f(DLDataTypeCode::kFloat, 64, 1, double, Double);                      \
    f(DLDataTypeCode::kBFloat, 16, 1, BFloat16, BFloat16);                 \
    f(DLDataTypeCode::kFloat8_e4m3fn, 8, 1, Float8_e4m3fn, Float8_e4m3fn); \
    f(DLDataTypeCode::kFloat8_e5m2, 8, 1, Float8_e5m2, Float8_e5m2);       \
    //f(DLDataTypeCode::kFloat8_e3m4, 8, 1, Float8_e3m4, Float8_e3m4);                      \
    //f(DLDataTypeCode::kFloat8_e4m3, 8, 1, Float8_e4m3, Float8_e4m3);                      \
    //f(DLDataTypeCode::kFloat8_e4m3b11fnuz, 8, 1, Float8_e4m3b11fnuz, Float8_e4m3b11fnuz); \
    // f(DLDataTypeCode::kFloat8_e4m3fnuz, 8, 1, Float8_e4m3fnuz,Float8_e4m3fnuz);    \
// f(DLDataTypeCode::kFloat8_e5m2fnuz, 8, 1, Float8_e5m2fnuz,Float8_e5m2fnuz);    \
// f(DLDataTypeCode::kFloat8_e8m0fnu, 8, 1, Float8_e8m0fnu,Float8_e8m0fnu);     \
// f(DLDataTypeCode::kFloat6_e2m3fn, 6, 1, Float6_e2m3fn,Float6_e2m3fn);      \
// f(DLDataTypeCode::kFloat6_e3m2fn, 6, 1, Float6_e3m2fn,Float6_e3m2fn);      \
// f(DLDataTypeCode::kFloat4_e2m1fn, 4, 1, Float4_e2m1fn,Float4_e2m1fn)


}// namespace aethermind

namespace std {

template<>
struct hash<aethermind::DataType> {
    NODISCARD static int cantor_pairing_function(int a, int b) { return (a + b) * (a + b + 1) / 2 + b; }
    std::size_t operator()(aethermind::DataType const& dtype) const noexcept {
        int a = static_cast<int>(dtype.code());
        int b = dtype.bits();
        int c = dtype.lanes();
        int d = cantor_pairing_function(a, b);
        return cantor_pairing_function(c, d);
    }
};

}// namespace std

#endif//AETHERMIND_DATA_TYPE_H
