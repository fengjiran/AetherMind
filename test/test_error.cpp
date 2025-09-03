//
// Created by richard on 9/3/25.
//
#include <gtest/gtest.h>

#include "error.h"

using namespace aethermind;

namespace {

TEST(AETHERMIND_THROW, BasicExceptionThrowing) {
    EXPECT_THROW({
        AETHERMIND_THROW(index_error) << "Test index out of bounds";
    }, Error);
}

}