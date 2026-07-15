#include "aethermind/base/tensor.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/memory/buffer.h"

#include "../test_utils/tensor_factory.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

using namespace aethermind;
using namespace aethermind::test_utils;

namespace {

// ── helpers ──────────────────────────────────────────────────────────

// Returns a non-null IntArrayView with size 0, suitable for rank-0.
inline IntArrayView EmptyIntArrayView() {
    static constexpr int64_t dummy = 0;
    return IntArrayView{&dummy, static_cast<size_t>(0)};
}

Tensor MakeRankZeroFloat32Tensor(size_t buffer_nbytes = 4) {
    auto buffer = aethermind::test_utils::detail::MakeBuffer(buffer_nbytes);
    ShapeAndStride sas;
    sas.set_contiguous(IntArrayView{});
    return {std::move(buffer), 0, DataType::Float32(), sas};
}

Tensor MakeShapeZeroTensor() {
    static constexpr std::array<int64_t, 1> kShape{0};
    static constexpr std::array<int64_t, 1> kStrides{1};
    auto buffer = aethermind::test_utils::detail::MakeBuffer(0);
    return {std::move(buffer), 0, DataType::Float32(),
            IntArrayView{kShape.data(), kShape.size()},
            IntArrayView{kStrides.data(), kStrides.size()}};
}

// ── characterization: existing non-rank-zero behavior ───────────────

TEST(TensorRankZero, DefaultTensorIsNotRankZero) {
    Tensor t;
    EXPECT_FALSE(t.is_initialized());
}

TEST(TensorRankZero, VectorShapeOneIsNotRankZero) {
    Tensor t = MakeContiguousTensor({1});
    EXPECT_TRUE(t.is_initialized());
    EXPECT_EQ(t.rank(), 1);
    EXPECT_NE(t.rank(), 0);
}

TEST(TensorRankZero, VectorShapeZeroIsNotRankZero) {
    Tensor t = MakeShapeZeroTensor();
    EXPECT_TRUE(t.is_initialized());
    EXPECT_EQ(t.rank(), 1);
    EXPECT_EQ(t.numel(), 0);
}

// ── rank-zero Tensor construction & metadata ────────────────────────

TEST(TensorRankZero, ConstructWithFourByteFloat32Buffer) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_TRUE(t.is_initialized());
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.itemsize(), 4U);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_NE(t.data(), nullptr);
    EXPECT_NE(t.mutable_data(), nullptr);
}

TEST(TensorRankZero, LogicalAndTouchedBytesEqualItemsize) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_EQ(t.logical_nbytes(), 4U);
    EXPECT_EQ(t.max_touched_span_bytes(), 4U);
    EXPECT_EQ(t.max_touched_element_offset(), 0);
}

TEST(TensorRankZero, ValidByteOffsetAndAlignment) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_EQ(t.byte_offset(), 0U);
    // alignment from posix_memalign with default alignment=64
    EXPECT_GT(t.alignment(), 0U);
    EXPECT_TRUE(t.storage_range_is_valid());
}

TEST(TensorRankZero, TooSmallBufferDeath) {
    EXPECT_DEATH(MakeRankZeroFloat32Tensor(0), "Check failed");
    EXPECT_DEATH(MakeRankZeroFloat32Tensor(3), "Check failed");
}

TEST(TensorRankZero, ShapeAndStridesAreEmpty) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_TRUE(t.shape().empty());
    EXPECT_TRUE(t.strides().empty());
    EXPECT_EQ(t.shape().size(), 0U);
    EXPECT_EQ(t.strides().size(), 0U);
}

// ── is_rank_zero predicate ─────────────────────────────────────────

TEST(TensorRankZero, IsRankZeroPredicate) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_TRUE(t.is_rank_zero());
}

TEST(TensorRankZero, DefaultTensorIsNotRankZeroViaPredicate) {
    Tensor t;
    EXPECT_FALSE(t.is_rank_zero());
}

TEST(TensorRankZero, VectorShapeOneIsNotRankZeroViaPredicate) {
    Tensor t = MakeContiguousTensor({1});
    EXPECT_FALSE(t.is_rank_zero());
}

TEST(TensorRankZero, VectorShapeZeroIsNotRankZeroViaPredicate) {
    Tensor t = MakeShapeZeroTensor();
    EXPECT_FALSE(t.is_rank_zero());
}

// ── immutable view borrow ──────────────────────────────────────────

TEST(TensorViewRankZero, ImmutableViewBorrowsRankZeroTensor) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    // Write a value through mutable access to verify later
    *static_cast<float*>(t.mutable_data()) = 3.14F;

    TensorView view = t.view();
    ASSERT_TRUE(view.is_valid());
    EXPECT_EQ(view.data(), t.data());
    EXPECT_EQ(view.dtype(), t.dtype());
    EXPECT_EQ(view.rank(), 0);
    EXPECT_EQ(view.numel(), 1);
    EXPECT_EQ(view.itemsize(), 4U);
    EXPECT_TRUE(view.is_contiguous());
    EXPECT_TRUE(view.is_rank_zero());
}

// ── mutable view borrow / write-through ────────────────────────────

TEST(MutableTensorViewRankZero, MutableViewWritesThroughTensorStorage) {
    Tensor t = MakeRankZeroFloat32Tensor(4);

    MutableTensorView view = t.mutable_view();
    ASSERT_TRUE(view.is_valid());
    ASSERT_NE(view.data(), nullptr);

    *view.data<float>() = 42.0F;
    EXPECT_FLOAT_EQ(*static_cast<float*>(t.mutable_data()), 42.0F);

    // Verify is_rank_zero on mutable view
    EXPECT_TRUE(view.is_rank_zero());
}

TEST(MutableTensorViewRankZero, MutableViewDefaultInvalid) {
    MutableTensorView view;
    EXPECT_FALSE(view.is_valid());
    EXPECT_FALSE(view.is_rank_zero());
}

// ── raw-part rank-0 views with null metadata pointers ───────────────

TEST(TensorViewRankZero, RawPartsNullMetadataPointersRankZero) {
    float data = 7.0F;
    TensorView view(&data,
                    DataType::Float32(),
                    EmptyIntArrayView(),
                    EmptyIntArrayView(),
                    0);

    ASSERT_TRUE(view.is_valid());
    EXPECT_EQ(view.rank(), 0);
    EXPECT_EQ(view.numel(), 1);
    EXPECT_TRUE(view.is_contiguous());
    EXPECT_TRUE(view.is_rank_zero());
}

TEST(TensorViewRankZero, RankZeroNullDataDeath) {
    EXPECT_DEATH(
            TensorView(nullptr,
                       DataType::Float32(),
                       EmptyIntArrayView(),
                       EmptyIntArrayView(),
                       0),
            "Check failed");
}

// ── [0]-shape: zero elements, null data is legal ──────────────────

TEST(TensorViewRankZero, ZeroShapeNullDataAccepted) {
    constexpr std::array<int64_t, 1> kShape{0};
    constexpr std::array<int64_t, 1> kStrides{1};

    TensorView view(nullptr,
                    DataType::Float32(),
                    IntArrayView{kShape.data(), kShape.size()},
                    IntArrayView{kStrides.data(), kStrides.size()},
                    0);

    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view.numel(), 0);
    EXPECT_FALSE(view.is_rank_zero());
}

// ── default views are not rank-zero ────────────────────────────────

TEST(TensorViewRankZero, DefaultViewIsNotRankZero) {
    TensorView view;
    EXPECT_FALSE(view.is_valid());
    EXPECT_FALSE(view.is_rank_zero());
}

// ── rank-0 dim/stride access death (DCHECK-based) ───────────────────

TEST(TensorRankZero, DimOutOfBoundsDeath) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(t.dim(0)), "Check failed");
#endif
}

TEST(TensorRankZero, StrideOutOfBoundsDeath) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(t.stride(0)), "Check failed");
#endif
}

// ── rank-0 slice/narrow rejection ─────────────────────────────────

TEST(TensorRankZero, SliceRejectsRankZero) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_DEATH(static_cast<void>(t.slice(0, 0, 1)), "Check failed");
}

TEST(TensorRankZero, NarrowRejectsRankZero) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    EXPECT_DEATH(static_cast<void>(t.narrow(0, 0, 1)), "Check failed");
}

// ── TensorView dim/stride rejection for rank-0 (DCHECK-based) ──────

TEST(TensorViewRankZero, DimOutOfBoundsDeath) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    TensorView view = t.view();
    ASSERT_TRUE(view.is_valid());
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(view.dim(0)), "Check failed");
#endif
}

TEST(TensorViewRankZero, StrideOutOfBoundsDeath) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    TensorView view = t.view();
    ASSERT_TRUE(view.is_valid());
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(view.stride(0)), "Check failed");
#endif
}

// ── MutableTensorView dim/stride rejection for rank-0 (DCHECK-based)

TEST(MutableTensorViewRankZero, DimOutOfBoundsDeath) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    MutableTensorView view = t.mutable_view();
    ASSERT_TRUE(view.is_valid());
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(view.dim(0)), "Check failed");
#endif
}

TEST(MutableTensorViewRankZero, StrideOutOfBoundsDeath) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    MutableTensorView view = t.mutable_view();
    ASSERT_TRUE(view.is_valid());
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(view.stride(0)), "Check failed");
#endif
}

// ── alignment test for rank-0 raw view ─────────────────────────────

TEST(TensorViewRankZero, ValidAlignmentPassedThrough) {
    alignas(64) float data = 1.0F;
    TensorView view(&data,
                    DataType::Float32(),
                    EmptyIntArrayView(),
                    EmptyIntArrayView(),
                    64);

    ASSERT_TRUE(view.is_valid());
    EXPECT_EQ(view.alignment(), 64U);
}

// ── MutableTensorView rank-0 raw parts ────────────────────────────

TEST(MutableTensorViewRankZero, RawPartsConstructionRankZero) {
    float storage = 0.0F;
    MutableTensorView view(&storage,
                           DataType::Float32(),
                           EmptyIntArrayView(),
                           EmptyIntArrayView(),
                           0);

    ASSERT_TRUE(view.is_valid());
    EXPECT_EQ(view.rank(), 0);
    EXPECT_EQ(view.numel(), 1);

    *view.data<float>() = 99.0F;
    EXPECT_FLOAT_EQ(storage, 99.0F);
}

TEST(MutableTensorViewRankZero, RawPartsNullDataDeath) {
    EXPECT_DEATH(
            MutableTensorView(nullptr,
                              DataType::Float32(),
                              EmptyIntArrayView(),
                              EmptyIntArrayView(),
                              0),
            "Check failed");
}

// ── mismatched shape/stride ranks reject ───────────────────────────

TEST(TensorViewRankZero, MismatchedShapeStrideRanksDeath) {
    constexpr std::array<int64_t, 1> kShape{1};
    float data;
    EXPECT_DEATH(
            TensorView(&data,
                       DataType::Float32(),
                       IntArrayView{kShape.data(), kShape.size()},
                       EmptyIntArrayView(),
                       0),
            "Check failed");
}

// ── logical nbytes / contiguous for rank-0 view ───────────────────

TEST(TensorViewRankZero, LogicalNBytesMatchesItemsize) {
    Tensor t = MakeRankZeroFloat32Tensor(4);
    TensorView view = t.view();
    EXPECT_EQ(view.logical_nbytes(), 4U);
    EXPECT_TRUE(view.is_contiguous());
}

// ── non-rank-zero positive-element null-data rejection ─────────────

TEST(TensorViewRankZero, NullDataShapeOneRejected) {
    constexpr std::array<int64_t, 1> kShape{1};
    constexpr std::array<int64_t, 1> kStrides{1};
    EXPECT_DEATH(
            TensorView(nullptr, DataType::Float32(),
                       IntArrayView{kShape.data(), kShape.size()},
                       IntArrayView{kStrides.data(), kStrides.size()}, 0),
            "Check failed");
}

TEST(TensorViewRankZero, NullDataShapeTwoByThreeRejected) {
    constexpr std::array<int64_t, 2> kShape{2, 3};
    constexpr std::array<int64_t, 2> kStrides{3, 1};
    EXPECT_DEATH(
            TensorView(nullptr, DataType::Float32(),
                       IntArrayView{kShape.data(), kShape.size()},
                       IntArrayView{kStrides.data(), kStrides.size()}, 0),
            "Check failed");
}

TEST(MutableTensorViewRankZero, NullDataShapeOneRejected) {
    constexpr std::array<int64_t, 1> kShape{1};
    constexpr std::array<int64_t, 1> kStrides{1};
    EXPECT_DEATH(
            MutableTensorView(nullptr, DataType::Float32(),
                              IntArrayView{kShape.data(), kShape.size()},
                              IntArrayView{kStrides.data(), kStrides.size()}, 0),
            "Check failed");
}

TEST(MutableTensorViewRankZero, NullDataShapeTwoByThreeRejected) {
    constexpr std::array<int64_t, 2> kShape{2, 3};
    constexpr std::array<int64_t, 2> kStrides{3, 1};
    EXPECT_DEATH(
            MutableTensorView(nullptr, DataType::Float32(),
                              IntArrayView{kShape.data(), kShape.size()},
                              IntArrayView{kStrides.data(), kStrides.size()}, 0),
            "Check failed");
}

// ── zero-element shapes with null data accepted ────────────────────

TEST(TensorViewRankZero, NullDataShapeTwoZeroThreeAccepted) {
    constexpr std::array<int64_t, 3> kShape{2, 0, 3};
    constexpr std::array<int64_t, 3> kStrides{3, 3, 1};
    TensorView view(nullptr, DataType::Float32(),
                    IntArrayView{kShape.data(), kShape.size()},
                    IntArrayView{kStrides.data(), kStrides.size()}, 0);
    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view.numel(), 0);
    EXPECT_FALSE(view.is_rank_zero());
}

TEST(MutableTensorViewRankZero, NullDataShapeTwoZeroThreeAccepted) {
    constexpr std::array<int64_t, 3> kShape{2, 0, 3};
    constexpr std::array<int64_t, 3> kStrides{3, 3, 1};
    MutableTensorView view(nullptr, DataType::Float32(),
                           IntArrayView{kShape.data(), kShape.size()},
                           IntArrayView{kStrides.data(), kStrides.size()}, 0);
    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view.numel(), 0);
    EXPECT_FALSE(view.is_rank_zero());
}

TEST(MutableTensorViewRankZero, NullDataShapeZeroAccepted) {
    constexpr std::array<int64_t, 1> kShape{0};
    constexpr std::array<int64_t, 1> kStrides{1};
    MutableTensorView view(nullptr, DataType::Float32(),
                           IntArrayView{kShape.data(), kShape.size()},
                           IntArrayView{kStrides.data(), kStrides.size()}, 0);
    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view.numel(), 0);
    EXPECT_FALSE(view.is_rank_zero());
}

}// namespace
