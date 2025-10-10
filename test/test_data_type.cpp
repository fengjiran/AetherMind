//
// Created by richard on 10/5/25.
//
#include "data_type.h"

#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace {

using namespace aethermind;

// 测试DataType的基本构造和访问方法
TEST(DataTypeTest, BasicConstruction) {
    // 默认构造函数
    DataType dtype_default;
    EXPECT_EQ(dtype_default.code(), DLDataTypeCode::Undefined);
    EXPECT_EQ(dtype_default.bits(), 0);
    EXPECT_EQ(dtype_default.lanes(), 0);

    // 从DLDataType构造
    DLDataType dl_dtype = {DLDataTypeCode::kFloat, 32, 1};
    DataType dtype_from_dl(dl_dtype);
    EXPECT_EQ(dtype_from_dl.code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(dtype_from_dl.bits(), 32);
    EXPECT_EQ(dtype_from_dl.lanes(), 1);

    // 从code、bits、lanes构造
    DataType dtype_manual(DLDataTypeCode::kInt, 64, 1);
    EXPECT_EQ(dtype_manual.code(), DLDataTypeCode::kInt);
    EXPECT_EQ(dtype_manual.bits(), 64);
    EXPECT_EQ(dtype_manual.lanes(), 1);

    // 类型转换运算符
    DLDataType converted = dtype_manual;
    EXPECT_EQ(converted.code, DLDataTypeCode::kInt);
    EXPECT_EQ(converted.bits, 64);
    EXPECT_EQ(converted.lanes, 1);
}

// 测试DataType的赋值运算符
TEST(DataTypeTest, AssignmentOperator) {
    DataType dtype1(DLDataTypeCode::kFloat, 32, 1);
    DataType dtype2;
    dtype2 = dtype1;
    EXPECT_EQ(dtype2.code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(dtype2.bits(), 32);
    EXPECT_EQ(dtype2.lanes(), 1);

    // 自赋值测试
    dtype1 = dtype1;
    EXPECT_EQ(dtype1.code(), DLDataTypeCode::kFloat);
}

// 测试DataType的类型检查方法
TEST(DataTypeTest, TypeChecks) {
    // 整数类型
    DataType int32 = DataType::Int(32);
    EXPECT_TRUE(int32.is_int());
    EXPECT_FALSE(int32.is_uint());
    EXPECT_FALSE(int32.is_float32());
    EXPECT_FALSE(int32.is_bool());

    // 无符号整数类型
    DataType uint8 = DataType::UInt(8);
    EXPECT_FALSE(uint8.is_int());
    EXPECT_TRUE(uint8.is_uint());
    EXPECT_FALSE(uint8.is_float32());

    // 布尔类型
    DataType boolean = DataType::Bool();
    EXPECT_TRUE(boolean.is_bool());
    EXPECT_FALSE(boolean.is_int());
    EXPECT_TRUE(boolean.is_uint(true));// Bool is implemented as UInt(1)

    // 浮点类型
    DataType float32 = DataType::Float(32);
    EXPECT_FALSE(float32.is_int());
    EXPECT_FALSE(float32.is_uint());
    EXPECT_TRUE(float32.is_float32());
    EXPECT_FALSE(float32.is_float16());
    EXPECT_FALSE(float32.is_bfloat16());

    // 半精度浮点类型
    DataType float16 = DataType::Float(16);
    EXPECT_TRUE(float16.is_float());
    EXPECT_TRUE(float16.is_float16());
    EXPECT_TRUE(float16.is_half());

    // BFloat16类型
    DataType bfloat16 = DataType::BFloat(16);
    EXPECT_TRUE(bfloat16.is_bfloat16());
    EXPECT_TRUE(bfloat16.is_float());

    // Float8类型变体
    DataType f8e4m3 = DataType::Float8E4M3();
    EXPECT_TRUE(f8e4m3.is_float8());
    EXPECT_TRUE(f8e4m3.is_float8_e4m3());
    EXPECT_FALSE(f8e4m3.is_float8_e5m2());

    DataType f8e5m2 = DataType::Float8E5M2();
    EXPECT_TRUE(f8e5m2.is_float8());
    EXPECT_FALSE(f8e5m2.is_float8_e4m3());
    EXPECT_TRUE(f8e5m2.is_float8_e5m2());

    // 句柄类型
    DataType handle = DataType::Handle();
    EXPECT_TRUE(handle.is_handle());
    EXPECT_FALSE(handle.is_void());

    // Void类型
    DataType void_type = DataType::Void();
    EXPECT_FALSE(void_type.is_handle());
    EXPECT_TRUE(void_type.is_void());

    // complex
    DataType complex_half = DataType::ComplexHalf();
    DataType complex_float = DataType::ComplexFloat();
    DataType complex_double = DataType::ComplexDouble();
    EXPECT_TRUE(complex_half.is_complex_half());
    EXPECT_TRUE(complex_float.is_complex_float());
    EXPECT_TRUE(complex_double.is_complex_double());
}

// 测试DataType的向量类型检查
TEST(DataTypeTest, VectorChecks) {
    // 标量类型
    DataType scalar = DataType::Float(32);
    EXPECT_TRUE(scalar.is_scalar());
    EXPECT_FALSE(scalar.is_vector());
    EXPECT_FALSE(scalar.is_fixed_length_vector());
    EXPECT_FALSE(scalar.is_scalable_vector());

    // 固定长度向量
    DataType fixed_vector = DataType::Float(32, 4);
    EXPECT_FALSE(fixed_vector.is_scalar());
    EXPECT_TRUE(fixed_vector.is_vector());
    EXPECT_TRUE(fixed_vector.is_fixed_length_vector());
    EXPECT_FALSE(fixed_vector.is_scalable_vector());
    EXPECT_TRUE(fixed_vector.is_scalable_or_fixed_length_vector());

    // 测试向量布尔类型
    DataType vector_bool = DataType::Bool(4);
    EXPECT_TRUE(vector_bool.is_vector_bool());
}

// 测试DataType的辅助方法
TEST(DataTypeTest, HelperMethods) {
    DataType float32 = DataType::Float(32);
    EXPECT_EQ(float32.nbytes(), 4);// 32 bits = 4 bytes

    DataType int8 = DataType::Int(8);
    EXPECT_EQ(int8.nbytes(), 1);// 8 bits = 1 byte

    DataType bool_type = DataType::Bool();
    EXPECT_EQ(bool_type.nbytes(), 1);// 1 bit is stored as 1 byte

    // 测试with_lanes方法
    DataType float32x4 = float32.with_lanes(4);
    EXPECT_EQ(float32x4.lanes(), 4);
    EXPECT_EQ(float32x4.code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(float32x4.bits(), 32);

    // 测试with_bits方法
    DataType float16 = float32.with_bits(16);
    EXPECT_EQ(float16.bits(), 16);
    EXPECT_EQ(float16.code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(float16.lanes(), 1);

    // 测试element_of方法
    DataType element = float32x4.element_of();
    EXPECT_EQ(element.lanes(), 1);
    EXPECT_EQ(element.code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(element.bits(), 32);
}

// 测试DataType的静态工厂方法
TEST(DataTypeTest, StaticFactoryMethods) {
    // 测试基本类型工厂方法
    EXPECT_EQ(DataType::Int(32).code(), DLDataTypeCode::kInt);
    EXPECT_EQ(DataType::Int(32).bits(), 32);
    EXPECT_EQ(DataType::Int(32).lanes(), 1);

    EXPECT_EQ(DataType::UInt(8).code(), DLDataTypeCode::kUInt);
    EXPECT_EQ(DataType::UInt(8).bits(), 8);
    EXPECT_EQ(DataType::UInt(8).lanes(), 1);

    EXPECT_EQ(DataType::Float(64).code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(DataType::Float(64).bits(), 64);

    EXPECT_TRUE(DataType::Bool().is_bool());

    // 测试特定类型快捷方法
    EXPECT_EQ(DataType::Float32().code(), DLDataTypeCode::kFloat);
    EXPECT_EQ(DataType::Float32().bits(), 32);

    // 测试Float8变体快捷方法
    EXPECT_TRUE(DataType::Float8E4M3().is_float8_e4m3());
    EXPECT_TRUE(DataType::Float8E5M2().is_float8_e5m2());
    EXPECT_TRUE(DataType::Float8E4M3FN().is_float8_e4m3fn());
    EXPECT_TRUE(DataType::Float8E5M2FNUZ().is_float8_e5m2fnuz());

    // 测试句柄类型
    EXPECT_TRUE(DataType::Handle().is_handle());
    EXPECT_TRUE(DataType::Void().is_void());
}

// 测试DataType的相等性比较
TEST(DataTypeTest, EqualityOperators) {
    DataType float32_1 = DataType::Float(32);
    DataType float32_2 = DataType::Float(32);
    DataType float64 = DataType::Float(64);
    DataType int32 = DataType::Int(32);
    DataType float32x4 = DataType::Float(32, 4);
    DataType undefined;
    DataType undefined2;

    // 相同类型比较
    EXPECT_TRUE(float32_1 == float32_2);
    EXPECT_FALSE(float32_1 != float32_2);

    // 不同类型比较
    EXPECT_FALSE(float32_1 == float64);
    EXPECT_TRUE(float32_1 != float64);

    // 不同代码类型比较
    EXPECT_FALSE(float32_1 == int32);
    EXPECT_TRUE(float32_1 != int32);

    // 不同lanes比较
    EXPECT_FALSE(float32_1 == float32x4);
    EXPECT_TRUE(float32_1 != float32x4);

    // 未定义类型比较
    EXPECT_TRUE(undefined == undefined2);
    EXPECT_FALSE(undefined == float32_1);
}

// 测试DataType的边界情况
TEST(DataTypeTest, EdgeCases) {
    // 测试空类型
    DataType void_type = DataType::Void();
    EXPECT_TRUE(void_type.is_void());
    EXPECT_EQ(void_type.bits(), 0);
    EXPECT_EQ(void_type.lanes(), 0);

    // 测试向量bool类型
    DataType vector_bool_2 = DataType::Bool(2);
    EXPECT_TRUE(vector_bool_2.is_vector_bool());
    EXPECT_EQ(vector_bool_2.lanes(), 2);

    // 测试多种浮点类型变体
    DataType f8_e3m4 = DataType::Float8E3M4();
    DataType f8_e4m3 = DataType::Float8E4M3();
    DataType f8_e5m2 = DataType::Float8E5M2();
    EXPECT_TRUE(f8_e3m4.is_float8());
    EXPECT_TRUE(f8_e4m3.is_float8());
    EXPECT_TRUE(f8_e5m2.is_float8());
    EXPECT_FALSE(f8_e3m4.is_float8_e4m3());
    EXPECT_FALSE(f8_e3m4.is_float8_e5m2());

    // 测试自定义lanes的类型
    DataType custom_lanes = DataType::Int(32, 8);
    EXPECT_FALSE(custom_lanes.is_scalar());
    EXPECT_TRUE(custom_lanes.is_vector());
    EXPECT_EQ(custom_lanes.lanes(), 8);
}

// 测试DataType的字符串转换
TEST(DataTypeTest, StringConversion) {
    // 测试DataTypeToString函数（假设已实现）
    DataType float32 = DataType::Float(32);
    std::string float32_str = DataTypeToString(float32);
    EXPECT_EQ(float32_str, "Float");

    // 测试流输出运算符
    std::stringstream ss;
    ss << float32;
    std::string streamed_str = ss.str();
    EXPECT_FALSE(streamed_str.empty());
}

// 测试DataTypeToString函数对基本标量类型的转换
TEST(DataTypeToStringTest, BasicScalarTypes) {
    // 测试布尔类型
    EXPECT_EQ(DataTypeToString(DataType::Bool()), "bool");

    // 测试有符号整数类型
    EXPECT_EQ(DataTypeToString(DataType::Int(8)), "Char");
    EXPECT_EQ(DataTypeToString(DataType::Int(16)), "Short");
    EXPECT_EQ(DataTypeToString(DataType::Int(32)), "Int");
    EXPECT_EQ(DataTypeToString(DataType::Int(64)), "Long");

    // 测试无符号整数类型
    EXPECT_EQ(DataTypeToString(DataType::UInt(8)), "Byte");
    EXPECT_EQ(DataTypeToString(DataType::UInt(16)), "UInt16");
    EXPECT_EQ(DataTypeToString(DataType::UInt(32)), "UInt32");
    EXPECT_EQ(DataTypeToString(DataType::UInt(64)), "UInt64");

    // 测试浮点类型
    EXPECT_EQ(DataTypeToString(DataType::Float(16)), "Half");
    EXPECT_EQ(DataTypeToString(DataType::Float(32)), "Float");
    EXPECT_EQ(DataTypeToString(DataType::Float(64)), "Double");

    // 测试BFloat16类型
    EXPECT_EQ(DataTypeToString(DataType::BFloat(16)), "BFloat16");

    // 测试Float8类型变体
    EXPECT_EQ(DataTypeToString(DataType::Float8E4M3FN()), "Float8_e4m3fn");
    EXPECT_EQ(DataTypeToString(DataType::Float8E5M2()), "Float8_e5m2");
}

// 测试DataTypeToString函数对void类型的转换
TEST(DataTypeToStringTest, VoidType) {
    EXPECT_EQ(DataTypeToString(DataType::Void()), "void");
}

// 测试DataTypeToString函数对固定长度向量类型的转换
TEST(DataTypeToStringTest, FixedLengthVectorTypes) {
    // 测试整数向量
    EXPECT_EQ(DataTypeToString(DataType::Int(32, 2)), "Intx2");
    EXPECT_EQ(DataTypeToString(DataType::Int(32, 4)), "Intx4");
    EXPECT_EQ(DataTypeToString(DataType::UInt(8, 8)), "Bytex8");

    // 测试浮点向量
    EXPECT_EQ(DataTypeToString(DataType::Float(32, 2)), "Floatx2");
    EXPECT_EQ(DataTypeToString(DataType::Float(32, 4)), "Floatx4");
    EXPECT_EQ(DataTypeToString(DataType::Float(64, 2)), "Doublex2");

    // 测试布尔向量
    EXPECT_EQ(DataTypeToString(DataType::Bool(2)), "Boolx2");
    EXPECT_EQ(DataTypeToString(DataType::Bool(4)), "Boolx4");
}

// 测试DataTypeToString函数对可缩放向量类型的转换
TEST(DataTypeToStringTest, ScalableVectorTypes) {
    // GTEST_SKIP();
    // 创建可缩放向量类型（通过直接设置lanes为负值）
    DataType scalable_int{DLDataTypeCode::kInt, 32, -4};
    EXPECT_EQ(DataTypeToString(scalable_int), "Intxvscalex4");

    DataType scalable_float(DLDataTypeCode::kFloat, 32, -8);
    EXPECT_EQ(DataTypeToString(scalable_float), "Floatxvscalex8");

    DataType scalable_bool(DLDataTypeCode::kUInt, 1, -2);
    EXPECT_EQ(DataTypeToString(scalable_bool), "Boolxvscalex2");
}

// 测试DataTypeToString函数对边界情况的处理
TEST(DataTypeToStringTest, EdgeCases) {
    // 测试默认构造的DataType
    DataType undefined;
    // 根据函数实现，未定义类型可能不会在SCALAR_TYPE_TO_CPP_TYPE_AND_NAME中匹配到
    // 因此结果可能是空字符串或其他值，这里使用EXPECT_FALSE来验证它不是任何已知类型
    EXPECT_FALSE(DataTypeToString(undefined) == "bool");
    EXPECT_FALSE(DataTypeToString(undefined) == "void");

    // 测试非标准但有效的组合
    DataType custom_int(DLDataTypeCode::kInt, 24, 1);
    // 由于24位整数不在SCALAR_TYPE_TO_CPP_TYPE_AND_NAME中定义，结果可能为空
    // 这里使用EXPECT_FALSE验证它不是标准名称
    EXPECT_FALSE(DataTypeToString(custom_int) == "Int");
    EXPECT_FALSE(DataTypeToString(custom_int) == "Char");
    EXPECT_FALSE(DataTypeToString(custom_int) == "Short");

    // 测试向量和可缩放向量的最大合理值
    DataType large_vector(DLDataTypeCode::kFloat, 32, 1024);
    EXPECT_TRUE(DataTypeToString(large_vector).find("Floatx") == 0);

    DataType large_scalable(DLDataTypeCode::kFloat, 32, -1024);
    EXPECT_TRUE(DataTypeToString(large_scalable).find("Floatxvscalex") == 0);
}

// 测试DataTypeToString函数与DataType的其他方法结合使用
TEST(DataTypeToStringTest, CombinedWithOtherMethods) {
    // 测试通过with_lanes方法创建的向量
    DataType float32 = DataType::Float(32);
    DataType float32x4 = float32.with_lanes(4);
    EXPECT_EQ(DataTypeToString(float32x4), "Floatx4");

    // 测试element_of方法获取元素类型
    DataType element = float32x4.element_of();
    EXPECT_EQ(DataTypeToString(element), "Float");

    // 测试不同位宽的转换
    DataType float16 = float32.with_bits(16);
    EXPECT_EQ(DataTypeToString(float16), "Half");
}

}// namespace