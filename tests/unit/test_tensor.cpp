//
// Created by 赵丹 on 25-6-17.
//
#include "env.h"

#include <gtest/gtest.h>

#if BUILD_WITH_TORCH
#include <torch/torch.h>
#endif

using namespace aethermind;
using namespace aethermind::details;

namespace {

TEST(Tensor, format) {
    std::string s1 = std::format("The answer is {}.", 42);
    EXPECT_EQ(s1, "The answer is 42.");
}

TEST(Tensor, envs) {
    EXPECT_TRUE(has_env("THP_MEM_ALLOC_ENABLE"));
    EXPECT_TRUE(check_env("THP_MEM_ALLOC_ENABLE"));

    EXPECT_TRUE(has_env("TRACEBACK_LIMIT"));
    auto trackback_limit = std::stoi(get_env("TRACEBACK_LIMIT").value());
    EXPECT_EQ(trackback_limit, 512);
}

}// namespace
