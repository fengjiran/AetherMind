//
// Created by 赵丹 on 2025/8/15.
//
#include "any.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Any, int) {
    Tensor t({3, 10});
    AetherMindAny a;
    TypeTraits<Tensor>::MoveToAny(std::move(t), &a);
    EXPECT_TRUE(a.tag_ == Tag::Tensor);
    auto t2 = TypeTraits<Tensor>::MoveFromAnyAfterCheck(&a);
    EXPECT_EQ(t2.use_count(), 1);
}

}