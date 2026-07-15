#include "test_const_eval_helpers.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

using namespace test_utils;

// ── MakeContiguousStrides helper tests ──

TEST(ConstEvaluator, MakeContiguousStridesEmpty) {
    const auto result = MakeContiguousStrides(std::span<const int64_t>{});
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->empty());
}

TEST(ConstEvaluator, MakeContiguousStridesZeroElement) {
    const std::vector<int64_t> shape{0, 3};
    const auto result = MakeContiguousStrides(shape);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 2U);
    EXPECT_EQ((*result)[1], 1);
    // strides[0] = 1 * 3 = 3 (no overflow since trailing dim is small)
    EXPECT_EQ((*result)[0], 3);
}

TEST(ConstEvaluator, MakeContiguousStridesOverflow) {
    constexpr int64_t kHuge = 1LL << 62;
    const std::vector<int64_t> shape{0, kHuge, kHuge};
    const auto result = MakeContiguousStrides(shape);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kOverflow);
}

}// namespace
}// namespace aethermind
