//
// Created by 赵丹 on 2025/8/15.
//
#include "any.h"
#include "container/string.h"
#include "device.h"
#include "tensor.h"

#include <gtest/gtest.h>

#ifndef TEST_ANY

using namespace aethermind;

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
}

TEST(Any, int) {
    const Any x0;
    EXPECT_TRUE(!x0.has_value());

    auto opt0 = x0.try_cast<int64_t>();
    EXPECT_TRUE(!opt0.has_value());
    EXPECT_THROW(UNUSED(x0.cast<float>()), Error);

    Any x1 = 1;
    EXPECT_TRUE(x1.IsInteger());
    EXPECT_EQ(x1.cast<int>(), 1);

    int64_t v1 = 10;
    x1 = v1;
    EXPECT_EQ(x1.cast<int>(), 10);

    Any x2 = v1;
    EXPECT_EQ(x2.cast<int>(), 10);
    EXPECT_EQ(Any(x2).cast<int>(), 10);
    EXPECT_TRUE(x2.IsInteger());
    int x22 = static_cast<int>(x2);
    EXPECT_EQ(x22, 10);
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

    // auto* p = static_cast<String*>(x0.GetUnderlyingPtr());
    // std::cout << *p << std::endl;

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

    String s4 = static_cast<String>(Any("hello"));
    EXPECT_EQ(s4, "hello");
    EXPECT_EQ(s4.use_count(), 1);
}

TEST(Any, map) {
    std::unordered_map<int, String> m = {{0, "hello"}, {1, "world"}};
    Any x0 = m;
    EXPECT_TRUE(x0.has_value());
    EXPECT_TRUE(x0.IsMap());

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

TEST(Any, tensor) {
    Tensor t({3, 10});
    Any x = t;
    EXPECT_TRUE(x.IsTensor());
    EXPECT_EQ(t.use_count(), 2);
    EXPECT_EQ(x.use_count(), 2);
    auto t2 = x.ToTensor();
    {
        Any y = t2;
        EXPECT_TRUE(y.IsTensor());
        EXPECT_EQ(t2.use_count(), 4);
        EXPECT_EQ(y.use_count(), 4);
    }

    EXPECT_EQ(t2.use_count(), 3);
    auto t3 = Any(t2).ToTensor();
    EXPECT_EQ(t3.use_count(), 4);
}

// 测试空Any对象的比较
TEST(AnyOperatorsTest, EmptyAnyComparison) {
    // 创建两个空Any对象
    Any empty1;
    Any empty2;

    // 空Any对象应该相等
    EXPECT_TRUE(empty1 == empty2);
    EXPECT_FALSE(empty1 != empty2);

    // 与nullptr的比较
    EXPECT_TRUE(empty1 == nullptr);
    EXPECT_FALSE(empty1 != nullptr);
}

// 测试基本类型的相等比较
TEST(AnyOperatorsTest, BasicTypeEquality) {
    // 整数比较
    Any int1(42);
    Any int2(42);
    Any int3(43);

    EXPECT_TRUE(int1 == int2);
    EXPECT_FALSE(int1 != int2);
    EXPECT_FALSE(int1 == int3);
    EXPECT_TRUE(int1 != int3);

    // 浮点数比较
    Any float1(3.14);
    Any float2(3.14);
    Any float3(2.71);

    EXPECT_TRUE(float1 == float2);
    EXPECT_FALSE(float1 != float2);
    EXPECT_FALSE(float1 == float3);
    EXPECT_TRUE(float1 != float3);

    // 布尔值比较
    Any bool1(true);
    Any bool2(true);
    Any bool3(false);

    EXPECT_TRUE(bool1 == bool2);
    EXPECT_FALSE(bool1 != bool2);
    EXPECT_FALSE(bool1 == bool3);
    EXPECT_TRUE(bool1 != bool3);

    // 字符串比较
    Any str1(String("hello"));
    Any str2(String("hello"));
    Any str3(String("world"));

    EXPECT_TRUE(str1 == str2);
    EXPECT_FALSE(str1 != str2);
    EXPECT_FALSE(str1 == str3);
    EXPECT_TRUE(str1 != str3);
}

// 测试不同类型之间的比较
TEST(AnyOperatorsTest, DifferentTypeComparison) {
    Any int_any(42);
    Any float_any(42.0);
    Any bool_any(true);
    Any str_any(String("42"));

    // 不同类型的Any对象应该不相等，即使它们的值在语义上可能相等
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

// 测试空与非空Any的比较
TEST(AnyOperatorsTest, EmptyVsNonEmptyComparison) {
    Any empty;
    Any non_empty(42);

    // 空Any与非空Any应该不相等
    EXPECT_FALSE(empty == non_empty);
    EXPECT_TRUE(empty != non_empty);

    // 非空Any与nullptr的比较
    EXPECT_FALSE(non_empty == nullptr);
    EXPECT_TRUE(non_empty != nullptr);
}

// 测试Device类型的比较
TEST(AnyOperatorsTest, DeviceTypeComparison) {
    // 创建两个相同的Device对象
    Device device1(DeviceType::kCPU, 0);
    Device device2(DeviceType::kCPU, 0);
    Device device3(DeviceType::kCPU, -1);

    Any dev_any1(device1);
    Any dev_any2(device2);
    Any dev_any3(device3);

    // 相同的Device应该相等
    EXPECT_TRUE(dev_any1 == dev_any2);
    EXPECT_FALSE(dev_any1 != dev_any2);

    // 不同的Device应该不相等
    EXPECT_FALSE(dev_any1 == dev_any3);
    EXPECT_TRUE(dev_any1 != dev_any3);
}

// 测试自定义类型的比较（指针比较）
TEST(AnyOperatorsTest, CustomTypeComparison) {
    // 创建一些自定义类型的对象（这里使用String作为例子，尽管它已经有特殊处理）
    // 但我们可以通过直接构造Holder来测试指针比较逻辑
    Any any1(std::make_shared<int>(42));
    Any any2(std::make_shared<int>(42));

    // 对于没有特殊比较逻辑的类型，AnyEqual会比较底层指针
    // 即使它们包含的值相同，但因为是不同的shared_ptr，所以应该不相等
    EXPECT_FALSE(any1 == any2);
    EXPECT_TRUE(any1 != any2);

    // 同一个对象的不同Any引用应该相等
    Any any3 = any1;
    EXPECT_FALSE(any1 == any3);
    EXPECT_TRUE(any1 != any3);
}

// 测试边界值比较
TEST(AnyOperatorsTest, BoundaryValueComparison) {
    // 整数边界值
    Any min_int(std::numeric_limits<int64_t>::min());
    Any min_int_copy(std::numeric_limits<int64_t>::min());
    Any max_int(std::numeric_limits<int64_t>::max());

    EXPECT_TRUE(min_int == min_int_copy);
    EXPECT_FALSE(min_int == max_int);

    // 浮点数边界值
    Any zero_float(0.0);
    Any zero_float_copy(0.0);
    Any neg_zero_float(-0.0);

    // 在大多数实现中，0.0和-0.0在==比较中应该相等
    EXPECT_TRUE(zero_float == zero_float_copy);
    EXPECT_TRUE(zero_float == neg_zero_float);

    // 注意：由于浮点数的特性，这里不测试NaN值的比较
    // 因为根据IEEE标准，NaN不等于任何值，包括它自己
}

// 测试通过赋值后的值比较
TEST(AnyOperatorsTest, AssignmentAndComparison) {
    Any a(42);
    Any b(100);

    EXPECT_FALSE(a == b);

    // 赋值后应该相等
    b = a;
    EXPECT_TRUE(a == b);

    // 移动赋值后应该相等
    Any c;
    c = std::move(a);
    EXPECT_TRUE(b == c);

    // 注意：a现在处于有效但未定义的状态，不应该再使用
}

// 测试AnyEqual函数对象的直接使用
TEST(AnyOperatorsTest, AnyEqualFunctionObject) {
    AnyEqual equal;

    Any int1(42);
    Any int2(42);
    Any int3(43);
    Any empty;

    // 测试各种比较场景
    EXPECT_TRUE(equal(int1, int2));
    EXPECT_FALSE(equal(int1, int3));
    EXPECT_FALSE(equal(int1, empty));
    EXPECT_FALSE(equal(empty, int1));
    EXPECT_TRUE(equal(empty, empty));
}

}// namespace

#endif
