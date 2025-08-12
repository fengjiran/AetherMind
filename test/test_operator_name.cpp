//
// Created by 赵丹 on 2025/8/12.
//

#include "operator_name.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(OpName, init) {
    OperatorName opname("aethermind::add", "");
    EXPECT_EQ(opname.name(), "aethermind::add");
    EXPECT_EQ(opname.overload_name(), "");
    EXPECT_EQ(opname.GetNamespace(), "aethermind");
    EXPECT_EQ(toString(opname), "aethermind::add");
    std::cout << opname << std::endl;

    EXPECT_TRUE(opname == OperatorName("aethermind::add", ""));
}

}