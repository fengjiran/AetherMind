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

TEST(ConstEvaluator, FoldingBudgetAcceptsExactLimits) {
    const ConstEvalPolicy policy{};
    const FoldingCost cost{
            .compute_ops = policy.max_compute_ops,
            .output_bytes = policy.max_output_bytes,
    };

    EXPECT_TRUE(CheckFoldingBudget(cost, policy).ok());
}

TEST(ConstEvaluator, FoldingBudgetRejectsComputeOverLimit) {
    const ConstEvalPolicy policy{};
    const FoldingCost cost{
            .compute_ops = static_cast<uint64_t>(policy.max_compute_ops) + 1U,
            .output_bytes = 0U,
    };

    EXPECT_EQ(CheckFoldingBudget(cost, policy).code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, FoldingBudgetRejectsOutputBytesOverLimit) {
    const ConstEvalPolicy policy{};
    const FoldingCost cost{
            .compute_ops = 0U,
            .output_bytes = policy.max_output_bytes + 1U,
    };

    EXPECT_EQ(CheckFoldingBudget(cost, policy).code(), StatusCode::kUnimplemented);
}

}// namespace
}// namespace aethermind
