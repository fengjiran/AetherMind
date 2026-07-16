#include "aethermind/shape_inference/broadcast.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace aethermind {
namespace {

// Helper: shorthand for SymbolicShape construction.
SymbolicShape Sym(std::initializer_list<ShapeSymbol> dims) {
    return {dims};
}

TEST(BroadcastShapes, SameShape) {
    std::vector<int64_t> lhs = {2, 3, 4};
    std::vector<int64_t> rhs = {2, 3, 4};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{2, 3, 4}));
}

TEST(BroadcastShapes, RankZero) {
    std::vector<int64_t> lhs = {};
    std::vector<int64_t> rhs = {};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->empty());
}

TEST(BroadcastShapes, RankZeroScalarBroadcast) {
    std::vector<int64_t> lhs = {};
    std::vector<int64_t> rhs = {3, 4};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{3, 4}));
}

TEST(BroadcastShapes, RankExpansion) {
    std::vector<int64_t> lhs = {4};
    std::vector<int64_t> rhs = {2, 3, 4};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{2, 3, 4}));
}

TEST(BroadcastShapes, BothSidedBroadcast) {
    std::vector<int64_t> lhs = {2, 1, 4};
    std::vector<int64_t> rhs = {1, 3, 4};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{2, 3, 4}));
}

TEST(BroadcastShapes, ZeroDimension) {
    // Zero dimensions are only broadcast-compatible with identical zero
    // or with 1 (the 1-lhs rule kicks in). Broadcasting 0 with non-zero
    // non-1 is incompatible.
    std::vector<int64_t> lhs = {0, 3};
    std::vector<int64_t> rhs = {0, 3};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{0, 3}));
}

TEST(BroadcastShapes, ZeroDimensionIncompatible) {
    std::vector<int64_t> lhs = {0, 3};
    std::vector<int64_t> rhs = {2, 3};
    auto result = BroadcastShapes(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(BroadcastShapes, OneDimension) {
    std::vector<int64_t> lhs = {1, 3};
    std::vector<int64_t> rhs = {2, 3};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{2, 3}));
}

TEST(BroadcastShapes, NegativeDimensionRejected) {
    std::vector<int64_t> lhs = {2, -1};
    std::vector<int64_t> rhs = {2, 3};
    auto result = BroadcastShapes(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(BroadcastShapes, IncompatibleDimensions) {
    std::vector<int64_t> lhs = {2, 5};
    std::vector<int64_t> rhs = {2, 3};
    auto result = BroadcastShapes(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(BroadcastShapes, BothRanksExpanded) {
    std::vector<int64_t> lhs = {1, 3};
    std::vector<int64_t> rhs = {2, 1, 3};
    auto result = BroadcastShapes(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{2, 1, 3}));
}

TEST(BroadcastInputStrides, SameShapeContiguous) {
    std::vector<int64_t> shape = {2, 3};
    std::vector<int64_t> strides = {3, 1};
    auto result = BroadcastInputStrides(shape, strides, shape);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, strides);
}

TEST(BroadcastInputStrides, StrideExpansion) {
    std::vector<int64_t> input_shape = {3};
    std::vector<int64_t> input_strides = {1};
    std::vector<int64_t> output_shape = {2, 3};
    auto result = BroadcastInputStrides(input_shape, input_strides, output_shape);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{0, 1}));
}

TEST(BroadcastInputStrides, BroadcastDimStrideZero) {
    std::vector<int64_t> input_shape = {1, 3};
    std::vector<int64_t> input_strides = {3, 1};
    std::vector<int64_t> output_shape = {2, 3};
    auto result = BroadcastInputStrides(input_shape, input_strides, output_shape);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, (std::vector<int64_t>{0, 1}));
}

TEST(BroadcastInputStrides, RankMismatch) {
    std::vector<int64_t> input_shape = {2, 3};
    std::vector<int64_t> input_strides = {3};
    std::vector<int64_t> output_shape = {2, 3};
    auto result = BroadcastInputStrides(input_shape, input_strides, output_shape);
    EXPECT_FALSE(result.ok());
}

TEST(BroadcastInputStrides, InputRankExceedsOutput) {
    std::vector<int64_t> input_shape = {2, 3, 4};
    std::vector<int64_t> input_strides = {12, 4, 1};
    std::vector<int64_t> output_shape = {3, 4};
    auto result = BroadcastInputStrides(input_shape, input_strides, output_shape);
    EXPECT_FALSE(result.ok());
}

TEST(BroadcastInputStrides, IncompatibleBroadcast) {
    std::vector<int64_t> input_shape = {2, 3};
    std::vector<int64_t> input_strides = {3, 1};
    std::vector<int64_t> output_shape = {4, 3};
    auto result = BroadcastInputStrides(input_shape, input_strides, output_shape);
    EXPECT_FALSE(result.ok());
}

TEST(InferBroadcastShape, FullyStaticSameShape) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(3),
                    ShapeSymbol::CreateFromValue(4)});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(3),
                    ShapeSymbol::CreateFromValue(4)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape, lhs);
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, FullyStaticRankZero) {
    auto lhs = Sym({});
    auto rhs = Sym({});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->output_shape.IsRankZero());
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, FullyStaticRankExpansion) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(4)});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(3),
                    ShapeSymbol::CreateFromValue(4)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape, rhs);
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, FullyStaticBothSidedBroadcast) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(1),
                    ShapeSymbol::CreateFromValue(4)});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(1),
                    ShapeSymbol::CreateFromValue(3),
                    ShapeSymbol::CreateFromValue(4)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(2),
                   ShapeSymbol::CreateFromValue(3),
                   ShapeSymbol::CreateFromValue(4)}));
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, FullyStaticZeroDimension) {
    // Zero dimensions are compatible only with identical zero or with 1.
    auto lhs = Sym({ShapeSymbol::CreateFromValue(0),
                    ShapeSymbol::CreateFromValue(3)});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(0),
                    ShapeSymbol::CreateFromValue(3)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape, lhs);
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, FullyStaticZeroDimensionIncompatible) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(0),
                    ShapeSymbol::CreateFromValue(3)});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(3)});
    auto result = InferBroadcastShape(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(InferBroadcastShape, IncompatibleStaticDimensions) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(5)});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(2),
                    ShapeSymbol::CreateFromValue(3)});
    auto result = InferBroadcastShape(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(InferBroadcastShape, SameSymbol) {
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({sym});
    auto rhs = Sym({sym});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape, lhs);
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, OnePlusSymbol) {
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({ShapeSymbol::CreateFromValue(1)});
    auto rhs = Sym({sym});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape, rhs);
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, StaticNPlusSymbol) {
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({ShapeSymbol::CreateFromValue(5)});
    auto rhs = Sym({sym});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(5)}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(InferBroadcastShape, SymbolPlusStaticN) {
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({sym});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(5)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(5)}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(InferBroadcastShape, DistinctSymbols) {
    auto sym1 = ShapeSymbol::Create();
    auto sym2 = ShapeSymbol::Create();
    auto lhs = Sym({sym1});
    auto rhs = Sym({sym2});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::Unknown()}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(InferBroadcastShape, SymbolPlusUnknown) {
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({sym});
    auto rhs = Sym({ShapeSymbol::Unknown()});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::Unknown()}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(InferBroadcastShape, BothUnknown) {
    auto lhs = Sym({ShapeSymbol::Unknown()});
    auto rhs = Sym({ShapeSymbol::Unknown()});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape, Sym({ShapeSymbol::Unknown()}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(InferBroadcastShape, StaticNPlusUnknown) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(7)});
    auto rhs = Sym({ShapeSymbol::Unknown()});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(7)}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(InferBroadcastShape, RankExpansionWithSymbolLeading) {
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({sym});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(2), sym});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(2), sym}));
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, MultiAxisStaticSymbolMix) {
    auto sym1 = ShapeSymbol::Create();
    auto sym2 = ShapeSymbol::Create();
    // lhs = [2, sym1, 4], rhs = [sym2, 3, 1]
    auto lhs = Sym({ShapeSymbol::CreateFromValue(2), sym1,
                    ShapeSymbol::CreateFromValue(4)});
    auto rhs = Sym({sym2, ShapeSymbol::CreateFromValue(3),
                    ShapeSymbol::CreateFromValue(1)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(2),
                   ShapeSymbol::CreateFromValue(3),
                   ShapeSymbol::CreateFromValue(4)}));
    ASSERT_EQ(result->deferred_axes.size(), 2U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[1].lhs_axis, 1U);
    EXPECT_EQ(result->deferred_axes[1].rhs_axis, 1U);
}

TEST(InferBroadcastShape, UnrankedLhsRejected) {
    SymbolicShape lhs;// unranked
    auto rhs = Sym({ShapeSymbol::CreateFromValue(2)});
    auto result = InferBroadcastShape(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(InferBroadcastShape, UnrankedRhsRejected) {
    auto lhs = Sym({ShapeSymbol::CreateFromValue(2)});
    SymbolicShape rhs;// unranked
    auto result = InferBroadcastShape(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(InferBroadcastShape, BothUnrankedRejected) {
    SymbolicShape lhs;
    SymbolicShape rhs;
    auto result = InferBroadcastShape(lhs, rhs);
    EXPECT_FALSE(result.ok());
}

TEST(InferBroadcastShape, DeterministicRepeatedInference) {
    auto sym1 = ShapeSymbol::Create();
    auto sym2 = ShapeSymbol::Create();
    auto lhs = Sym({ShapeSymbol::CreateFromValue(2), sym1});
    auto rhs = Sym({sym2, ShapeSymbol::CreateFromValue(3)});

    auto result1 = InferBroadcastShape(lhs, rhs);
    auto result2 = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result1.ok());
    ASSERT_TRUE(result2.ok());
    EXPECT_EQ(result1->output_shape, result2->output_shape);
    EXPECT_EQ(result1->deferred_axes, result2->deferred_axes);
}

TEST(InferBroadcastShape, DeterministicNoFreshSymbols) {
    // Repeated inference must not create fresh symbols.
    auto sym = ShapeSymbol::Create();
    auto lhs = Sym({sym});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(5)});

    auto result1 = InferBroadcastShape(lhs, rhs);
    auto result2 = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result1.ok());
    ASSERT_TRUE(result2.ok());
    EXPECT_EQ(result1->output_shape, result2->output_shape);
}

TEST(InferBroadcastShape, DeferredAxesInAscendingOutputOrder) {
    auto sym1 = ShapeSymbol::Create();
    auto sym2 = ShapeSymbol::Create();
    // lhs = [sym1, 5, sym2], rhs = [3, sym2, 7]
    auto lhs = Sym({sym1, ShapeSymbol::CreateFromValue(5), sym2});
    auto rhs = Sym({ShapeSymbol::CreateFromValue(3), sym2,
                    ShapeSymbol::CreateFromValue(7)});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(3),
                   ShapeSymbol::CreateFromValue(5),
                   ShapeSymbol::CreateFromValue(7)}));
    ASSERT_EQ(result->deferred_axes.size(), 3U);
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
    EXPECT_EQ(result->deferred_axes[1].lhs_axis, 1U);
    EXPECT_EQ(result->deferred_axes[1].rhs_axis, 1U);
    EXPECT_EQ(result->deferred_axes[2].lhs_axis, 2U);
    EXPECT_EQ(result->deferred_axes[2].rhs_axis, 2U);
}

TEST(InferBroadcastShape, DeferredAxesWithRankExpansion) {
    auto sym = ShapeSymbol::Create();
    // lhs = [3, sym], rhs = [sym]
    auto lhs = Sym({ShapeSymbol::CreateFromValue(3), sym});
    auto rhs = Sym({sym});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // output_rank = 2, lhs_axis_offset = 0, rhs_axis_offset = 1
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(3), sym}));
    EXPECT_TRUE(result->deferred_axes.empty());
}

TEST(InferBroadcastShape, DeferredAxesWithRankExpansionDistinct) {
    auto sym1 = ShapeSymbol::Create();
    auto sym2 = ShapeSymbol::Create();
    // lhs = [3, sym1], rhs = [sym2]
    auto lhs = Sym({ShapeSymbol::CreateFromValue(3), sym1});
    auto rhs = Sym({sym2});
    auto result = InferBroadcastShape(lhs, rhs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // output_rank = 2, lhs_axis_offset = 0, rhs_axis_offset = 1
    EXPECT_EQ(result->output_shape,
              Sym({ShapeSymbol::CreateFromValue(3),
                   ShapeSymbol::Unknown()}));
    ASSERT_EQ(result->deferred_axes.size(), 1U);
    // Deferred axes refer to original indices: lhs_axis=1, rhs_axis=0
    EXPECT_EQ(result->deferred_axes[0].lhs_axis, 1U);
    EXPECT_EQ(result->deferred_axes[0].rhs_axis, 0U);
}

TEST(BroadcastRoundTrip, ConcreteBroadcastThenStrides) {
    std::vector<int64_t> lhs_shape = {2, 1, 4};
    std::vector<int64_t> rhs_shape = {1, 3, 4};

    auto out_shape = BroadcastShapes(lhs_shape, rhs_shape);
    ASSERT_TRUE(out_shape.ok()) << out_shape.status().ToString();
    EXPECT_EQ(*out_shape, (std::vector<int64_t>{2, 3, 4}));

    std::vector<int64_t> lhs_strides = {4, 4, 1};
    auto lhs_eff = BroadcastInputStrides(lhs_shape, lhs_strides, *out_shape);
    ASSERT_TRUE(lhs_eff.ok()) << lhs_eff.status().ToString();
    EXPECT_EQ(*lhs_eff, (std::vector<int64_t>{4, 0, 1}));

    std::vector<int64_t> rhs_strides = {12, 4, 1};
    auto rhs_eff = BroadcastInputStrides(rhs_shape, rhs_strides, *out_shape);
    ASSERT_TRUE(rhs_eff.ok()) << rhs_eff.status().ToString();
    EXPECT_EQ(*rhs_eff, (std::vector<int64_t>{0, 4, 1}));
}

}// namespace
}// namespace aethermind
