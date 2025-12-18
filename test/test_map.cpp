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
}

}// namespace