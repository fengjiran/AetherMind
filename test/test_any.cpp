//
// Created by 赵丹 on 2025/8/15.
//
#include "any.h"
#include "testing_object.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Any, bool) {
    Any x0;
    auto opt0 = x0.as<bool>();
    EXPECT_TRUE(!opt0.has_value());

    EXPECT_THROW(
            {
                try {
                    MAYBE_UNUSED auto v0 = x0.cast<bool>();
                } catch (const Error&) {
                    throw;
                }
            },
            Error);

    Any x1 = true;
    EXPECT_TRUE(x1.cast<bool>());
    auto v1 = x1.cast<int>();
    EXPECT_EQ(v1, 1);
    x1 = false;
    EXPECT_TRUE(!x1.cast<bool>());
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
    Any x0;
    EXPECT_TRUE(x0.tag() == AnyTag::None);

    auto opt0 = x0.as<int64_t>();
    EXPECT_TRUE(!opt0.has_value());

    EXPECT_THROW(
            {
                try {
                    MAYBE_UNUSED auto v0 = x0.cast<float>();
                } catch (const Error&) {
                    throw;
                }
            },
            Error);

    Any x1 = 1;
    EXPECT_TRUE(x1.tag() == AnyTag::Int);
    EXPECT_EQ(x1.cast<int>(), 1);

    int64_t v1 = 10;
    x1 = v1;
    EXPECT_EQ(x1.cast<int>(), 10);
}

TEST(Any, float) {
    Any x0;
    auto opt0 = x0.as<double>();
    EXPECT_TRUE(!opt0.has_value());

    x0 = 1;
    auto v1 = x0.cast<float>();
    EXPECT_EQ(v1, 1);

    x0 = 2.2;
    auto v2 = x0.cast<float>();
    EXPECT_FLOAT_EQ(v2, 2.2);
}

TEST(Any, string) {
    Any x0 = "hello";
    EXPECT_EQ(x0.use_count(), 1);
    EXPECT_TRUE(x0.tag() == AnyTag::String);
    EXPECT_TRUE(x0.as<std::string>().has_value());
    EXPECT_EQ(x0.cast<std::string>(), "hello");

    x0 = std::string("world");
    EXPECT_EQ(x0.use_count(), 1);
    EXPECT_TRUE(x0.tag() == AnyTag::String);
    EXPECT_EQ(x0.cast<std::string>(), "world");
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
    EXPECT_TRUE(opt_v3.has_value());
    EXPECT_TRUE(*opt_v3);

    auto opt_v4 = x0.try_cast<double>();
    EXPECT_TRUE(opt_v4.has_value());
    EXPECT_EQ(opt_v4.value(), 1);

    Any x1 = true;
    auto opt_v5 = x1.as<bool>();
    EXPECT_TRUE(opt_v5.has_value());
    EXPECT_EQ(opt_v5.value(), 1);

    auto opt_v6 = x1.try_cast<int>();
    EXPECT_TRUE(opt_v6.has_value());
    EXPECT_EQ(opt_v6.value(), 1);

    auto opt_v7 = x1.try_cast<double>();
    EXPECT_TRUE(opt_v7.has_value());
}

TEST(Any, device) {
    Any x = Device(kCUDA, 1);
    auto dev = x.cast<Device>();
    EXPECT_TRUE(x.tag() == AnyTag::Device);
    EXPECT_EQ(dev.type(), kCUDA);
    EXPECT_EQ(dev.index(), 1);
}

TEST(Any, tensor) {
    Tensor t({3, 10});
    AetherMindAny a;
    TypeTraits<Tensor>::MoveToAny(std::move(t), &a);
    EXPECT_TRUE(a.tag_ == AnyTag::Tensor);
    auto t2 = TypeTraits<Tensor>::MoveFromAnyAfterCheck(&a);
    EXPECT_EQ(t2.use_count(), 1);

    {
        Any x = t2;
        EXPECT_TRUE(x.is_tensor());
        EXPECT_EQ(t2.use_count(), 2);
        EXPECT_EQ(x.use_count(), 2);
    }

    EXPECT_EQ(t2.use_count(), 1);
}

}// namespace