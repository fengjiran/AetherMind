//
// Created by 赵丹 on 2025/9/2.
//
#include "container/array.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Array, init) {
    Array<int> vec1(5, 42);
    EXPECT_TRUE(vec1.defined());
    EXPECT_TRUE(vec1.unique());
    EXPECT_EQ(vec1.size(), 5);
    EXPECT_EQ(vec1.capacity(), 5);
    EXPECT_EQ(vec1.front(), 42);
    EXPECT_EQ(*vec1.begin(), 42);
    EXPECT_EQ(vec1.back(), 42);
    EXPECT_EQ(*(vec1.end() - 1), 42);
}

}