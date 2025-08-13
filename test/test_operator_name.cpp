//
// Created by 赵丹 on 2025/8/12.
//

#include "operator_name.h"
#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(OpName, init) {
    OperatorName opname1("aethermind::add", "Tensor");
    OperatorName opname2("aethermind::add", "Scalar");

    EXPECT_EQ(opname1.name(), "aethermind::add");
    EXPECT_EQ(opname1.overload_name(), "Tensor");
    EXPECT_EQ(opname1.GetNamespace(), "aethermind");
    EXPECT_EQ(toString(opname1), "aethermind::add.Tensor");
    // std::cout << opname1 << std::endl;

    EXPECT_TRUE(opname1 == OperatorName("aethermind::add", "Tensor"));
    EXPECT_TRUE(opname1 != opname2);
}

}