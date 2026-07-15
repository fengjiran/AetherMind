#include "aethermind/base/shape_and_stride.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace aethermind;

namespace {

// ===== Baseline: default state =====
TEST(ShapeAndStrideRankZero, DefaultRemainsUninitialized) {
    ShapeAndStride ss;
    EXPECT_FALSE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 0);
    EXPECT_EQ(ss.numel(), 0);
    EXPECT_FALSE(ss.is_contiguous());
    EXPECT_EQ(ss.max_element_offset(), 0);
    EXPECT_EQ(ss.shape().size(), 0u);
    EXPECT_EQ(ss.strides().size(), 0u);
}

// ===== Baseline: existing rank-1 [N] behavior =====
TEST(ShapeAndStrideRankZero, VectorShapeOnePositive) {
    ShapeAndStride ss;
    ss.set_contiguous(IntArrayView(std::vector<int64_t>{5}));
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 1);
    EXPECT_EQ(ss.numel(), 5);
    EXPECT_TRUE(ss.is_contiguous());
    EXPECT_EQ(ss.dim(0), 5);
    EXPECT_EQ(ss.stride(0), 1);
    EXPECT_EQ(ss.max_element_offset(), 4);
}

// ===== Baseline: existing rank-1 [0] behavior (numel 0, contiguous) =====
TEST(ShapeAndStrideRankZero, VectorShapeZeroNumelZero) {
    ShapeAndStride ss;
    ss.set_contiguous(IntArrayView(std::vector<int64_t>{0}));
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 1);
    EXPECT_EQ(ss.numel(), 0);
    EXPECT_EQ(ss.max_element_offset(), 0);
    EXPECT_EQ(ss.dim(0), 0);
    EXPECT_EQ(ss.stride(0), 1);
}

// ===== Rank-zero: explicit empty shape produces scalar metadata =====

TEST(ShapeAndStrideRankZero, ExplicitEmptyShapeIsRankZero) {
    ShapeAndStride ss;
    ss.set_contiguous({});
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 0);
    EXPECT_EQ(ss.numel(), 1);
    EXPECT_TRUE(ss.is_contiguous());
    EXPECT_EQ(ss.max_element_offset(), 0);
    EXPECT_EQ(ss.shape().size(), 0u);
    EXPECT_EQ(ss.strides().size(), 0u);
}

TEST(ShapeAndStrideRankZero, ExplicitEmptyShapeViaSet) {
    ShapeAndStride ss;
    ss.set({}, {});
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 0);
    EXPECT_EQ(ss.numel(), 1);
    EXPECT_TRUE(ss.is_contiguous());
    EXPECT_EQ(ss.shape().size(), 0u);
    EXPECT_EQ(ss.strides().size(), 0u);
}

TEST(ShapeAndStrideRankZero, ConstructorEmptyIsRankZero) {
    ShapeAndStride ss({}, {});
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 0);
    EXPECT_EQ(ss.numel(), 1);
    EXPECT_TRUE(ss.is_contiguous());
}

// ===== Repeated set =====

TEST(ShapeAndStrideRankZero, RepeatedSetToRankZero) {
    ShapeAndStride ss;
    ss.set_contiguous(IntArrayView(std::vector<int64_t>{3}));
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.numel(), 3);

    ss.set_contiguous({});
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 0);
    EXPECT_EQ(ss.numel(), 1);
    EXPECT_TRUE(ss.is_contiguous());

    ss.set_contiguous(IntArrayView(std::vector<int64_t>{4, 5}));
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.size(), 2);
    EXPECT_EQ(ss.numel(), 20);
    EXPECT_TRUE(ss.is_contiguous());
}

TEST(ShapeAndStrideRankZero, RepeatedSetViaSetMethod) {
    ShapeAndStride ss;
    ss.set(IntArrayView(std::vector<int64_t>{2}), IntArrayView(std::vector<int64_t>{1}));
    EXPECT_TRUE(ss.is_initialized());

    ss.set({}, {});
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.numel(), 1);

    ss.set(IntArrayView(std::vector<int64_t>{7}), IntArrayView(std::vector<int64_t>{1}));
    EXPECT_TRUE(ss.is_initialized());
    EXPECT_EQ(ss.numel(), 7);
}

// ===== Death tests: invalid metadata =====

TEST(ShapeAndStrideRankZero, MismatchedShapeStrideDeath) {
    ShapeAndStride ss;
    EXPECT_DEATH(
            ss.set(IntArrayView(std::vector<int64_t>{2, 3}),
                   IntArrayView(std::vector<int64_t>{1})),
            "Check failed");
}

TEST(ShapeAndStrideRankZero, NegativeDimensionsDeath) {
    ShapeAndStride ss;
    EXPECT_DEATH(
            ss.set_contiguous(IntArrayView(std::vector<int64_t>{-1})),
            "Check failed");
}

TEST(ShapeAndStrideRankZero, NumelOverflowDeath) {
    ShapeAndStride ss;
    // Shape where each intermediate stride fits but the numel product overflows
    ss.set_contiguous(IntArrayView(std::vector<int64_t>{
            int64_t(1) << 20, int64_t(1) << 20, int64_t(1) << 20, int64_t(1) << 20}));
    EXPECT_DEATH(static_cast<void>(ss.numel()), "Check failed");
}

TEST(ShapeAndStrideRankZero, RankZeroDimDeath) {
    ShapeAndStride ss({}, {});
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(ss.dim(0)), "Check failed");
#endif
}

TEST(ShapeAndStrideRankZero, RankZeroStrideDeath) {
    ShapeAndStride ss({}, {});
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(ss.stride(0)), "Check failed");
#endif
}

}// namespace
