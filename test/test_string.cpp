//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(String, empty) {
    std::string s = "";
    String s2 = s;
    EXPECT_TRUE(s2.defined());
    EXPECT_TRUE(s2.empty());
}

}// namespace