//
// Created by 赵丹 on 2025/8/15.
//
#include "aethermind/base/tensor.h"
#include "any.h"
#include "container/array_view.h"
#include "container/string.h"
#include "device.h"
#include "test_utils/tensor_factory.h"
#include "test_utils/tensor_random.h"

#include <gtest/gtest.h>
#include <ranges>

using namespace aethermind;
using namespace aethermind::test_utils;

namespace {

TEST(Any, bool) {
    Any x0;
    EXPECT_EQ(x0.use_count(), 0);
    auto opt0 = x0.as<bool>();
    EXPECT_TRUE(!opt0.has_value());

    EXPECT_THROW(UNUSED(x0.cast<float>()), Error);
    EXPECT_THROW(UNUSED(x0.cast<bool>()), Error);

    Any x1 = true;
    EXPECT_TRUE(x1.cast<bool>());
    x1 = false;
    EXPECT_TRUE(!x1.cast<bool>());

    EXPECT_EQ(details::Type2Str<bool>::value(), "Bool");
    EXPECT_EQ(details::Type2Str<Tensor>::value(), "Tensor");
}

TEST(Any, null) {
    Any x0;
    EXPECT_TRUE(x0 == nullptr);
    EXPECT_FALSE(x0 != nullptr);

    x0 = 1;
    EXPECT_TRUE(x0 != nullptr);
    EXPECT_FALSE(x0 == nullptr);

    Any x1 = x0;
    EXPECT_TRUE(x1 != nullptr);
    EXPECT_FALSE(x1 == nullptr);

    x1 = nullptr;
    EXPECT_TRUE(x1 == nullptr);
    EXPECT_FALSE(x1 != nullptr);

    Any x2 = nullptr;
    EXPECT_TRUE(x2 == nullptr);
}

TEST(Any, int) {
    const Any x0;
    EXPECT_TRUE(!x0.has_value());

    auto opt0 = x0.try_cast<int64_t>();
    EXPECT_TRUE(!opt0.has_value());
    EXPECT_THROW(UNUSED(x0.cast<float>()), Error);

    Any x1 = 1;
    EXPECT_TRUE(x1.IsInteger());
    EXPECT_TRUE(x1.IsSmallObject());
    EXPECT_EQ(x1.cast<int>(), 1);

    int64_t v1 = 10;
    x1 = v1;
    EXPECT_EQ(x1.cast<int>(), 10);

    Any x2 = v1;
    EXPECT_EQ(x2.cast<int>(), 10);
    EXPECT_EQ(Any(x2).cast<int>(), 10);
    EXPECT_TRUE(x2.IsInteger());
    int x22 = x2;
    EXPECT_EQ(x22, 10);

    Any x3 = 100;
    int x33 = x3.cast<const int>();
    EXPECT_EQ(x33, 100);
}

TEST(Any, float) {
    Any x0;
    auto opt0 = x0.as<double>();
    EXPECT_TRUE(!opt0.has_value());

    x0 = 1;
    auto v1 = x0.cast<int>();
    EXPECT_EQ(v1, 1);

    x0 = 2.2;
    auto v2 = x0.cast<float>();
    EXPECT_FLOAT_EQ(v2, 2.2);

    float v3 = 3.14f;
    Any x3 = v3;
    EXPECT_EQ(x3.cast<float>(), 3.14f);
    EXPECT_TRUE(x3.unique());

    Any x2 = v3;
    EXPECT_EQ(x2.cast<float>(), 3.14f);
    EXPECT_EQ(Any(std::complex<float>(1, 2)).cast<std::complex<float>>().real(), 1.0f);
}

TEST(Any, string) {
    Any x0 = "hello";
    EXPECT_EQ(x0.use_count(), 1);
    EXPECT_TRUE(x0.IsString());
    EXPECT_TRUE(x0.as<String>().has_value());
    EXPECT_EQ(x0.ToString(), "hello");

    x0 = std::string("world");
    EXPECT_EQ(x0.use_count(), 1);
    EXPECT_TRUE(x0.IsString());
    EXPECT_EQ(x0.ToString(), "world");

    Any s0 = String("hello");
    Any s1 = "hello";
    Any s2 = std::string("hello");
    EXPECT_TRUE(s0.IsString());
    EXPECT_TRUE(s1.IsString());
    EXPECT_TRUE(s2.IsString());

    Any s3 = s0;
    EXPECT_EQ(s3.use_count(), 1);
    s3.reset();
    EXPECT_TRUE(s0.unique());

    String s4 = Any("hello");
    EXPECT_EQ(s4, "hello");
    EXPECT_EQ(s4.use_count(), 1);
}

TEST(Any, map) {
    std::unordered_map<int, String> m = {{0, "hello"}, {1, "world"}};
    Any x0 = m;
    EXPECT_TRUE(x0.has_value());
    EXPECT_TRUE(x0.IsMap());
    auto& s = x0.operator[]<std::unordered_map<int, String>>(0);
    EXPECT_EQ(s, "hello");

    auto y = x0.cast<std::unordered_map<int, String>>();
    EXPECT_TRUE(y[0] == "hello");
    EXPECT_TRUE(y[1] == "world");

    Any x1 = "hello";
    std::unordered_map<Any, int> b = {{x1, 1}};
    EXPECT_EQ(b[x1], 1);
}

TEST(Any, cast_vs_as) {
    Any x0 = 1;
    auto opt_v0 = x0.as<int64_t>();
    EXPECT_TRUE(opt_v0.has_value());
    EXPECT_EQ(*opt_v0, 1);

    auto opt_v1 = x0.as<bool>();
    EXPECT_TRUE(!opt_v1.has_value());

    auto opt_v2 = x0.as<float>();
    EXPECT_TRUE(!opt_v2.has_value());

    auto opt_v3 = x0.try_cast<bool>();
    EXPECT_TRUE(!opt_v3.has_value());

    auto opt_v4 = x0.try_cast<double>();
    EXPECT_TRUE(!opt_v4.has_value());

    Any x1 = true;
    auto opt_v5 = x1.as<bool>();
    EXPECT_TRUE(opt_v5.has_value());
    EXPECT_EQ(opt_v5.value(), 1);

    auto opt_v6 = x1.try_cast<int>();
    EXPECT_TRUE(!opt_v6.has_value());

    auto opt_v7 = x1.try_cast<double>();
    EXPECT_TRUE(!opt_v7.has_value());
}

TEST(Any, device) {
    Any x = Device(kCUDA, 1);
    auto dev = x.ToDevice();
    EXPECT_TRUE(x.IsDevice());
    EXPECT_EQ(dev.type(), kCUDA);
    EXPECT_EQ(dev.index(), 1);
}

TEST(Any, new_tensor) {
    Tensor t = MakeContiguousTensor({3, 10}, DataType::Float32());
    Any x = t;
    EXPECT_TRUE(x.IsNewTensor());
    
    Tensor t2 = x.ToNewTensor();
    EXPECT_EQ(t2.dtype(), t.dtype());
    EXPECT_TRUE(std::ranges::equal(t2.shape(), t.shape()));
    EXPECT_EQ(t2.numel(), t.numel());
    
    {
        Any y = t2;
        EXPECT_TRUE(y.IsNewTensor());
    }
}

TEST(Any, tensor_type_roundtrip) {
    Tensor new_tensor = RandomUniformTensor({5, 5}, DataType::Float32());
    Any any_new = new_tensor;
    
    EXPECT_TRUE(any_new.IsNewTensor());
    
    Tensor recovered_new = any_new.ToNewTensor();
    EXPECT_TRUE(recovered_new.is_initialized());
    EXPECT_TRUE(std::ranges::equal(recovered_new.shape(), new_tensor.shape()));
}

TEST(AnyOperatorsTest, EmptyAnyComparison) {
    Any empty1;
    Any empty2;

    EXPECT_TRUE(empty1 == empty2);
    EXPECT_FALSE(empty1 != empty2);

    EXPECT_TRUE(empty1 == nullptr);
    EXPECT_FALSE(empty1 != nullptr);
}

TEST(AnyOperatorsTest, BasicTypeEquality) {
    Any int1(42);
    Any int2(42);
    Any int3(43);

    EXPECT_TRUE(int1 == int2);
    EXPECT_FALSE(int1 != int2);
    EXPECT_FALSE(int1 == int3);
    EXPECT_TRUE(int1 != int3);

    Any float1(3.14);
    Any float2(3.14);
    Any float3(2.71);

    EXPECT_TRUE(float1 == float2);
    EXPECT_FALSE(float1 != float2);
    EXPECT_FALSE(float1 == float3);
    EXPECT_TRUE(float1 != float3);

    Any bool1(true);
    Any bool2(true);
    Any bool3(false);

    EXPECT_TRUE(bool1 == bool2);
    EXPECT_FALSE(bool1 != bool2);
    EXPECT_FALSE(bool1 == bool3);
    EXPECT_TRUE(bool1 != bool3);

    Any str1(String("hello"));
    Any str2(String("hello"));
    Any str3(String("world"));

    EXPECT_TRUE(str1 == str2);
    EXPECT_FALSE(str1 != str2);
    EXPECT_FALSE(str1 == str3);
    EXPECT_TRUE(str1 != str3);
}

TEST(AnyOperatorsTest, DifferentTypeComparison) {
    Any int_any(42);
    Any float_any(42.0);
    Any bool_any(true);
    Any str_any(String("42"));

    EXPECT_FALSE(int_any == float_any);
    EXPECT_TRUE(int_any != float_any);

    EXPECT_FALSE(int_any == bool_any);
    EXPECT_TRUE(int_any != bool_any);

    EXPECT_FALSE(int_any == str_any);
    EXPECT_TRUE(int_any != str_any);

    EXPECT_FALSE(float_any == bool_any);
    EXPECT_TRUE(float_any != bool_any);

    EXPECT_FALSE(float_any == str_any);
    EXPECT_TRUE(float_any != str_any);

    EXPECT_FALSE(bool_any == str_any);
    EXPECT_TRUE(bool_any != str_any);
}

TEST(AnyOperatorsTest, EmptyVsNonEmptyComparison) {
    Any empty;
    Any non_empty(42);

    EXPECT_FALSE(empty == non_empty);
    EXPECT_TRUE(empty != non_empty);

    EXPECT_FALSE(non_empty == nullptr);
    EXPECT_TRUE(non_empty != nullptr);
}

TEST(AnyOperatorsTest, DeviceTypeComparison) {
    Device device1(DeviceType::kCPU, 0);
    Device device2(DeviceType::kCPU, 0);
    Device device3(DeviceType::kCPU, -1);

    Any dev_any1(device1);
    Any dev_any2(device2);
    Any dev_any3(device3);

    EXPECT_TRUE(dev_any1 == dev_any2);
    EXPECT_FALSE(dev_any1 != dev_any2);

    EXPECT_FALSE(dev_any1 == dev_any3);
    EXPECT_TRUE(dev_any1 != dev_any3);
}

TEST(AnyOperatorsTest, BoundaryValueComparison) {
    Any min_int(std::numeric_limits<int64_t>::min());
    Any min_int_copy(std::numeric_limits<int64_t>::min());
    Any max_int(std::numeric_limits<int64_t>::max());

    EXPECT_TRUE(min_int == min_int_copy);
    EXPECT_FALSE(min_int == max_int);

    Any zero_float(0.0);
    Any zero_float_copy(0.0);
    Any neg_zero_float(-0.0);

    EXPECT_TRUE(zero_float == zero_float_copy);
    EXPECT_TRUE(zero_float == neg_zero_float);
}

TEST(AnyOperatorsTest, AssignmentAndComparison) {
    Any a(42);
    Any b(100);

    EXPECT_FALSE(a == b);

    b = a;
    EXPECT_TRUE(a == b);

    Any c;
    c = std::move(a);
    EXPECT_TRUE(b == c);
}

TEST(AnyOperatorsTest, AnyEqualFunctionObject) {
    AnyEqual equal;

    Any int1(42);
    Any int2(42);
    Any int3(43);
    Any empty;

    EXPECT_TRUE(equal(int1, int2));
    EXPECT_FALSE(equal(int1, int3));
    EXPECT_FALSE(equal(int1, empty));
    EXPECT_FALSE(equal(empty, int1));
    EXPECT_TRUE(equal(empty, empty));
}

bool TestPrintOutput(const Any& any, AnyPrintFormat format, const String& expected) {
    std::ostringstream oss;
    PrintAny(oss, any, format);
    String result = oss.str();

    if (result != expected) {
        std::cerr << "Test failed! Format: " << static_cast<int>(format) << "\n";
        std::cerr << "Expected: \"" << expected << "\"\n";
        std::cerr << "Got:      \"" << result << "\"\n\n";
        return false;
    }
    return true;
}

TEST(AnyPrintTest, TestBasicTypes) {
    std::cout << "=== Testing Basic Types ===\n";

    Any any_int(42);
    TestPrintOutput(any_int, AnyPrintFormat::Default, "42");
    TestPrintOutput(any_int, AnyPrintFormat::Compact, "42");

    Any any_float(3.14159);
    TestPrintOutput(any_float, AnyPrintFormat::Default, "3.14159");

    Any any_bool(true);
    TestPrintOutput(any_bool, AnyPrintFormat::Default, "1");

    Any any_string("Hello, World!");
    TestPrintOutput(any_string, AnyPrintFormat::Default, "Hello, World!");

    Any any_cstring("C-style string");
    TestPrintOutput(any_cstring, AnyPrintFormat::Default, "C-style string");
}

TEST(AnyPrintTest, TestPrintFormats) {
    std::cout << "=== Testing Print Formats ===\n";

    Any any_int(42);

    TestPrintOutput(any_int, AnyPrintFormat::Default, "42");
    TestPrintOutput(any_int, AnyPrintFormat::Compact, "42");

    std::ostringstream oss;
    PrintAny(oss, any_int, AnyPrintFormat::Debug);
    String debug_output = oss.str();
    std::cout << "Debug format output: " << debug_output << "\n";

    if (debug_output.find("int") != String::npos && debug_output.find("42") != String::npos) {
        std::cout << "✓ Debug format contains type and value\n";
    } else {
        std::cout << "✗ Debug format missing type or value\n";
    }
}

TEST(AnyPrintTest, TestEmptyAny) {
    std::cout << "\n=== Testing Empty Any ===\n";

    Any any_empty;

    std::ostringstream oss_default, oss_debug, oss_compact;
    PrintAny(oss_default, any_empty, AnyPrintFormat::Default);
    PrintAny(oss_debug, any_empty, AnyPrintFormat::Debug);
    PrintAny(oss_compact, any_empty, AnyPrintFormat::Compact);

    std::cout << "Default format: " << oss_default.str() << "\n";
    std::cout << "Debug format: " << oss_debug.str() << "\n";
    std::cout << "Compact format: " << oss_compact.str() << "\n";
}

TEST(AnyPrintTest, TestOperatorStream) {
    std::cout << "\n=== Testing operator<< ===\n";

    Any any_int(42);
    Any any_string("Test");

    std::ostringstream oss1, oss2;
    oss1 << any_int;
    oss2 << any_string;

    std::cout << "operator<<(42): " << oss1.str() << "\n";
    std::cout << "operator<<(\"Test\"): " << oss2.str() << "\n";

    std::ostringstream oss3, oss4;
    PrintAny(oss3, any_int, AnyPrintFormat::Default);
    PrintAny(oss4, any_string, AnyPrintFormat::Default);

    if (oss1.str() == oss3.str()) {
        std::cout << "✓ operator<< matches PrintAny default format for int\n";
    }

    if (oss2.str() == oss4.str()) {
        std::cout << "✓ operator<< matches PrintAny default format for string\n";
    }
}

TEST(AnyPrintTest, TestCustomType) {
    std::cout << "\n=== Testing Custom Type ===\n";

    struct Point {
        int x;
        int y;
    };

    Point p = {10, 20};
    Any any_point(p);

    std::ostringstream oss;
    PrintAny(oss, any_point, AnyPrintFormat::Default);
    String output = oss.str();
    std::cout << "Custom type output: " << output << "\n";

    if (output.find("Point") != String::npos || output.find("@") != String::npos) {
        std::cout << "✓ Custom type output contains expected information\n";
    }
}

}// namespace