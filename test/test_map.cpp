//
// Created by richard on 12/18/25.
//
#include "container/map.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(MAPTEST, basic) {
    Map<int, int> dict;
    EXPECT_TRUE(dict.empty());
    EXPECT_EQ(dict.slots(), 2);
    dict.insert(1, 2);
    dict.insert(2, 3);
    EXPECT_EQ(dict.size(), 2);
    EXPECT_TRUE(dict.IsSmallMap());

    EXPECT_EQ(dict[1], 2);
    EXPECT_EQ(dict[2], 3);

    dict.insert(3, 4);
    dict.insert(4, 5);
    EXPECT_EQ(dict.size(), 4);
    EXPECT_TRUE(dict.IsSmallMap());
    EXPECT_EQ(dict.slots(), 4);

    dict.insert(5, 6);
    dict.insert(6, 7);
    EXPECT_EQ(dict.size(), 6);
    EXPECT_TRUE(!dict.IsSmallMap());
    EXPECT_EQ(dict.slots(), 8);
}

}// namespace