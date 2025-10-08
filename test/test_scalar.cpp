//
// Created by richard on 10/8/25.
//
#include "scalar.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(Scalar, init) {
    Scalar s1 = false;
    EXPECT_EQ(s1.toBool(), false);
    EXPECT_TRUE(s1.type() == DataType::Bool());
    EXPECT_TRUE(s1.is_bool());
    s1 = true;
    std::cout << s1 << std::endl;
    std::cout << toString(s1) << std::endl;

    Scalar s2 = 10;
    EXPECT_EQ(s2.toInt(), 10);
    EXPECT_TRUE(s2.is_integral());
    std::cout << s2 << std::endl;
    std::cout << toString(s2) << std::endl;

    Scalar s3 = 1.5;
    EXPECT_EQ(s3.toFloat(), 1.5);
    EXPECT_TRUE(s3.is_floating_point());
    EXPECT_TRUE(std::isfinite(s3.toFloat()));
    std::cout << s3 << std::endl;
    std::cout << toString(s3) << std::endl;
}

}// namespace