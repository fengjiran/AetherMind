//
// Created by 赵丹 on 2025/8/24.
//
#include "storage.h"

#include <gtest/gtest.h>

using namespace aethermind;
namespace {

TEST(Storage, init) {
    Storage s1;
    EXPECT_EQ(s1.use_count(), 0);
    EXPECT_FALSE(s1.defined());
    EXPECT_EQ(s1.nbytes(), 0);
    EXPECT_TRUE(s1.data() == nullptr);
    EXPECT_TRUE(s1.device() == Device(kUndefined));
}

}