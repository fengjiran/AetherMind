//
// Created by 赵丹 on 2025/8/15.
//
#include "any.h"
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
    EXPECT_TRUE(x0.tag() == Tag::None);

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
    EXPECT_TRUE(x1.tag() == Tag::Int);
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

TEST(Any, device) {
    Any x = Device(kCUDA, 1);
    auto dev = x.cast<Device>();
    EXPECT_TRUE(x.tag() == Tag::Device);
    EXPECT_EQ(dev.type(), kCUDA);
    EXPECT_EQ(dev.index(), 1);
}

TEST(Any, tensor) {
    Tensor t({3, 10});
    AetherMindAny a;
    TypeTraits<Tensor>::MoveToAny(std::move(t), &a);
    EXPECT_TRUE(a.tag_ == Tag::Tensor);
    auto t2 = TypeTraits<Tensor>::MoveFromAnyAfterCheck(&a);
    EXPECT_EQ(t2.use_count(), 1);
}

}// namespace