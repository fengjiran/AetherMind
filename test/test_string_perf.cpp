//
// Created by richard on 11/28/25.
//
import test_module;

#include "container/string.h"
#include "test_utils.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(ModuleTest, basic) {
    auto x = add(1, 2);
    EXPECT_EQ(3, x);
}

}