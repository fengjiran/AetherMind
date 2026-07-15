// Created by richard on 12/18/25.

#include "container/map_v2.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

TEST(MapImplV2Test, basic) {
    MapImplV2<int, int> map_impl(32);
    EXPECT_EQ(map_impl.slots(), 32);
    std::pair<int, int> x{1, 10};
    map_impl.insert(x);
    auto it = map_impl.begin();
    map_impl.erase(it);
}

}// namespace
