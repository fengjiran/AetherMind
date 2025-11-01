//
// Created by 赵丹 on 2025/8/15.
//
#include "any.h"
#include "container/string.h"
#include "device.h"
#include "tensor.h"

#include <gtest/gtest.h>

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
    int x22 = x2;
    EXPECT_EQ(x22, 10);

    auto* p = static_cast<int64_t*>(x2.GetUnderlyingPtr());
    *p = 20;
    EXPECT_EQ(x2.cast<int>(), 20);
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
    EXPECT_EQ(s3.use_count(), 2);
    s3.reset();
    EXPECT_TRUE(s0.unique());

    String s4 = Any("hello");
    EXPECT_EQ(s4, "hello");
    EXPECT_EQ(s4.use_count(), 1);
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

}// namespace