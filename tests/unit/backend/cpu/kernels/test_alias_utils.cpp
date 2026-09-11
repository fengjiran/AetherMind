#include "aethermind/backend/cpu/kernels/common/alias_utils.h"

#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using namespace aethermind::cpu::detail;

TEST(CpuAliasUtils, RangesOverlapUsesHalfOpenIntervals) {
    EXPECT_TRUE(RangesOverlap(AddressRange{.begin = 100, .end = 110},
                              AddressRange{.begin = 105, .end = 115}));
    EXPECT_FALSE(RangesOverlap(AddressRange{.begin = 100, .end = 110},
                               AddressRange{.begin = 110, .end = 120}));
    EXPECT_FALSE(RangesOverlap(AddressRange{.begin = 100, .end = 100},
                               AddressRange{.begin = 90, .end = 110}));
}

TEST(CpuAliasUtils, ContiguousRangesSeparateAdjacentSlicesOfOneAllocation) {
    std::array<float, 8> storage{};
    const auto first_half = BuildContiguousAddressRange(
            storage.data(), 4, sizeof(float), "first half");
    const auto second_half = BuildContiguousAddressRange(
            storage.data() + 4, 4, sizeof(float), "second half");
    const auto shifted_tail = BuildContiguousAddressRange(
            storage.data() + 3, 4, sizeof(float), "shifted tail");

    ASSERT_TRUE(first_half.ok()) << first_half.status().ToString();
    ASSERT_TRUE(second_half.ok()) << second_half.status().ToString();
    ASSERT_TRUE(shifted_tail.ok()) << shifted_tail.status().ToString();
    EXPECT_EQ(first_half->end - first_half->begin, 4 * sizeof(float));
    EXPECT_FALSE(RangesOverlap(*first_half, *second_half));
    EXPECT_TRUE(RangesOverlap(*first_half, *shifted_tail));
}

TEST(CpuAliasUtils, BuildContiguousAddressRangeAcceptsEmptyViewWithoutData) {
    const auto empty = BuildContiguousAddressRange(nullptr, 0, sizeof(float), "empty view");

    ASSERT_TRUE(empty.ok()) << empty.status().ToString();
    EXPECT_EQ(empty->begin, empty->end);
    EXPECT_FALSE(RangesOverlap(*empty, AddressRange{.begin = 0, .end = 64}));
}

TEST(CpuAliasUtils, BuildContiguousAddressRangeRejectsInvalidGeometry) {
    std::array<float, 4> storage{};
    const auto null_data = BuildContiguousAddressRange(nullptr, 4, sizeof(float), "null data");
    const auto zero_item_size = BuildContiguousAddressRange(
            storage.data(), 4, 0, "zero item size");
    const auto negative_count = BuildContiguousAddressRange(
            storage.data(), -1, sizeof(float), "negative count");

    ASSERT_FALSE(null_data.ok());
    EXPECT_EQ(null_data.status().message(),
              "null data has invalid contiguous address geometry");
    ASSERT_FALSE(zero_item_size.ok());
    EXPECT_EQ(zero_item_size.status().message(),
              "zero item size has invalid contiguous address geometry");
    ASSERT_FALSE(negative_count.ok());
    EXPECT_EQ(negative_count.status().message(),
              "negative count has invalid contiguous address geometry");
}

TEST(CpuAliasUtils, BuildContiguousAddressRangeRejectsAddressOverflow) {
    const auto near_address_limit = reinterpret_cast<const void*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    const auto range = BuildContiguousAddressRange(
            near_address_limit, 2, sizeof(float), "test range");

    ASSERT_FALSE(range.ok());
    EXPECT_EQ(range.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(range.status().message(), "test range address range overflow");
}

TEST(CpuAliasUtils, RowwiseLayoutsCanProveDisjointSeparateStorage) {
    std::array<float, 4> lhs_storage{};
    std::array<float, 4> rhs_storage{};
    const auto lhs = BuildRowwiseAddressLayout(
            lhs_storage.data(), 2, 2, 2, 1, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressLayout(
            rhs_storage.data(), 2, 2, 2, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseLayoutOverlap(*lhs, *rhs), RowwiseLayoutOverlap::kDisjoint);
    EXPECT_FALSE(RowwiseLayoutsMayOverlap(*lhs, *rhs));
}

TEST(CpuAliasUtils, RowwiseLayoutsProveActualOverlapWithoutHoles) {
    std::array<float, 8> storage{};
    const auto lhs = BuildRowwiseAddressLayout(
            storage.data(), 2, 2, 4, 1, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressLayout(
            storage.data() + 1, 2, 2, 4, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseLayoutOverlap(*lhs, *rhs), RowwiseLayoutOverlap::kProvenOverlap);
    EXPECT_TRUE(RowwiseLayoutsMayOverlap(*lhs, *rhs));
}

TEST(CpuAliasUtils, RowwiseLayoutsProveSharedAllocationRowsAreDisjoint) {
    std::array<float, 8> storage{};
    const auto lhs = BuildRowwiseAddressLayout(
            storage.data(), 2, 2, 4, 1, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressLayout(
            storage.data() + 2, 2, 2, 4, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseLayoutOverlap(*lhs, *rhs), RowwiseLayoutOverlap::kDisjoint);
    EXPECT_FALSE(RowwiseLayoutsMayOverlap(*lhs, *rhs));
}

TEST(CpuAliasUtils, RowwiseLayoutsReportMayOverlapForColumnStrideHoles) {
    std::array<float, 8> storage{};
    const auto lhs = BuildRowwiseAddressLayout(
            storage.data(), 1, 4, 8, 2, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressLayout(
            storage.data() + 1, 1, 4, 8, 2, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseLayoutOverlap(*lhs, *rhs), RowwiseLayoutOverlap::kMayOverlap);
    EXPECT_TRUE(RowwiseLayoutsMayOverlap(*lhs, *rhs));
}

TEST(CpuAliasUtils, BuildRowwiseAddressLayoutRejectsAddressOverflow) {
    const auto near_address_limit = reinterpret_cast<const void*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    const auto layout = BuildRowwiseAddressLayout(
            near_address_limit, 1, 2, 2, 1, sizeof(float), "test layout");

    ASSERT_FALSE(layout.ok());
    EXPECT_EQ(layout.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuAliasUtils, ValidateNoRowwiseOverlapAcceptsProvablyDisjointLayouts) {
    std::array<float, 4> output_storage{};
    std::array<float, 4> input_storage{};
    const auto output = BuildRowwiseAddressLayout(
            output_storage.data(), 2, 2, 2, 1, sizeof(float), "output");
    const auto input = BuildRowwiseAddressLayout(
            input_storage.data(), 2, 2, 2, 1, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateNoRowwiseOverlap(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CpuAliasUtils, ValidateNoRowwiseOverlapRejectsProvenOverlap) {
    std::array<float, 8> storage{};
    const auto output = BuildRowwiseAddressLayout(
            storage.data() + 1, 2, 2, 4, 1, sizeof(float), "output");
    const auto input = BuildRowwiseAddressLayout(
            storage.data(), 2, 2, 4, 1, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateNoRowwiseOverlap(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "TestKernel output must not overlap input");
}

TEST(CpuAliasUtils, ValidateNoRowwiseOverlapReportsUndecidableHolesAsUnimplemented) {
    std::array<float, 8> storage{};
    const auto output = BuildRowwiseAddressLayout(
            storage.data() + 1, 1, 4, 8, 2, sizeof(float), "output");
    const auto input = BuildRowwiseAddressLayout(
            storage.data(), 1, 4, 8, 2, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateNoRowwiseOverlap(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
    EXPECT_EQ(status.message(),
              "TestKernel cannot prove output is disjoint from input "
              "for the requested strided layouts");
}

TEST(CpuAliasUtils, ClassifyLayoutInjectivitySeparatesProofFromUndecidable) {
    constexpr int64_t colliding_shape[2] = {2, 2};
    constexpr int64_t colliding_strides[2] = {1, 1};
    constexpr int64_t irregular_shape[2] = {2, 3};
    constexpr int64_t irregular_strides[2] = {3, 2};
    constexpr int64_t row_major_strides[2] = {3, 1};
    constexpr int64_t transposed_strides[2] = {1, 2};
    constexpr int64_t padded_strides[2] = {4, 1};
    constexpr int64_t broadcast_shape[1] = {2};
    constexpr int64_t broadcast_strides[1] = {0};
    constexpr int64_t unit_axis_shape[2] = {1, 3};
    constexpr int64_t unit_axis_strides[2] = {0, 1};

    // Offsets {0, 1, 1, 2} collide, and a repeated element does too.
    EXPECT_EQ(ClassifyLayoutInjectivity(colliding_shape, colliding_strides),
              LayoutInjectivity::kProvenNonInjective);
    EXPECT_EQ(ClassifyLayoutInjectivity(broadcast_shape, broadcast_strides),
              LayoutInjectivity::kProvenNonInjective);
    // Offsets {0, 2, 4, 3, 5, 7} are distinct, but the cheap proof cannot see it.
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, irregular_strides),
              LayoutInjectivity::kUnknown);
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, row_major_strides),
              LayoutInjectivity::kProvenInjective);
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, transposed_strides),
              LayoutInjectivity::kProvenInjective);
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, padded_strides),
              LayoutInjectivity::kProvenInjective);
    // Extent-1 axes never affect injectivity, whatever their stride.
    EXPECT_EQ(ClassifyLayoutInjectivity(unit_axis_shape, unit_axis_strides),
              LayoutInjectivity::kProvenInjective);
}

TEST(CpuAliasUtils, FootprintMarksTransposedLayoutDenseAndPaddedLayoutSparse) {
    std::array<float, 7> storage{};
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t transposed_strides[2] = {1, 2};
    constexpr int64_t padded_strides[2] = {4, 1};

    const auto transposed = BuildStridedAddressFootprint(
            storage.data(), shape, transposed_strides, sizeof(float), "transposed");
    const auto padded = BuildStridedAddressFootprint(
            storage.data(), shape, padded_strides, sizeof(float), "padded");

    ASSERT_TRUE(transposed.ok()) << transposed.status().ToString();
    ASSERT_TRUE(padded.ok()) << padded.status().ToString();
    EXPECT_EQ(transposed->numel, 6);
    EXPECT_EQ(transposed->max_offset, 5);
    EXPECT_TRUE(transposed->is_dense);
    EXPECT_FALSE(transposed->is_empty);
    EXPECT_EQ(padded->max_offset, 6);
    EXPECT_FALSE(padded->is_dense);
}

TEST(CpuAliasUtils, FootprintTreatsZeroExtentAsEmptyAndRankZeroAsOneElement) {
    std::array<float, 2> storage{};
    constexpr int64_t empty_shape[2] = {2, 0};
    constexpr int64_t empty_strides[2] = {1, 1};

    const auto empty = BuildStridedAddressFootprint(
            storage.data(), empty_shape, empty_strides, sizeof(float), "empty");
    const auto scalar = BuildStridedAddressFootprint(
            storage.data(), {}, {}, sizeof(float), "scalar");

    ASSERT_TRUE(empty.ok()) << empty.status().ToString();
    ASSERT_TRUE(scalar.ok()) << scalar.status().ToString();
    EXPECT_TRUE(empty->is_empty);
    EXPECT_EQ(empty->envelope.begin, empty->envelope.end);
    EXPECT_FALSE(scalar->is_empty);
    EXPECT_EQ(scalar->numel, 1);
    EXPECT_EQ(scalar->max_offset, 0);
    EXPECT_TRUE(scalar->is_dense);
    EXPECT_EQ(scalar->envelope.end - scalar->envelope.begin, sizeof(float));
}

TEST(CpuAliasUtils, FootprintOverlapDistinguishesProvenFromUndecidable) {
    std::array<float, 8> storage{};
    constexpr int64_t dense_shape[2] = {2, 2};
    constexpr int64_t dense_strides[2] = {2, 1};
    constexpr int64_t holed_shape[1] = {4};
    constexpr int64_t holed_strides[1] = {2};
    constexpr int64_t empty_shape[2] = {2, 0};
    constexpr int64_t empty_strides[2] = {2, 1};

    const auto dense_base = BuildStridedAddressFootprint(
            storage.data(), dense_shape, dense_strides, sizeof(float), "dense base");
    const auto dense_shifted = BuildStridedAddressFootprint(
            storage.data() + 1, dense_shape, dense_strides, sizeof(float), "dense shifted");
    const auto dense_far = BuildStridedAddressFootprint(
            storage.data() + 7, dense_shape, dense_strides, sizeof(float), "dense far");
    const auto holed_base = BuildStridedAddressFootprint(
            storage.data(), holed_shape, holed_strides, sizeof(float), "holed base");
    const auto holed_shifted = BuildStridedAddressFootprint(
            storage.data() + 1, holed_shape, holed_strides, sizeof(float), "holed shifted");
    const auto empty_at_base = BuildStridedAddressFootprint(
            storage.data(), empty_shape, empty_strides, sizeof(float), "empty at base");

    ASSERT_TRUE(dense_base.ok()) << dense_base.status().ToString();
    ASSERT_TRUE(dense_shifted.ok()) << dense_shifted.status().ToString();
    ASSERT_TRUE(dense_far.ok()) << dense_far.status().ToString();
    ASSERT_TRUE(holed_base.ok()) << holed_base.status().ToString();
    ASSERT_TRUE(holed_shifted.ok()) << holed_shifted.status().ToString();
    ASSERT_TRUE(empty_at_base.ok()) << empty_at_base.status().ToString();

    EXPECT_EQ(ClassifyFootprintOverlap(*dense_base, *dense_far), FootprintOverlap::kDisjoint);
    EXPECT_EQ(ClassifyFootprintOverlap(*dense_base, *dense_shifted),
              FootprintOverlap::kProvenOverlap);
    // A shared base address proves overlap even when neither view is dense.
    EXPECT_EQ(ClassifyFootprintOverlap(*holed_base, *dense_base),
              FootprintOverlap::kProvenOverlap);
    EXPECT_EQ(ClassifyFootprintOverlap(*holed_base, *holed_shifted),
              FootprintOverlap::kMayOverlap);
    EXPECT_EQ(ClassifyFootprintOverlap(*holed_base, *dense_shifted),
              FootprintOverlap::kMayOverlap);
    // An empty view never overlaps, not even at a shared base address.
    EXPECT_EQ(ClassifyFootprintOverlap(*dense_base, *empty_at_base),
              FootprintOverlap::kDisjoint);
}

TEST(CpuAliasUtils, ValidateInjectiveLayoutMapsProofOntoStatus) {
    const Status injective = ValidateInjectiveLayout(
            "TestKernel", LayoutInjectivity::kProvenInjective, "output");
    EXPECT_TRUE(injective.ok()) << injective.ToString();

    const Status collision = ValidateInjectiveLayout(
            "TestKernel", LayoutInjectivity::kProvenNonInjective, "output");
    EXPECT_EQ(collision.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(collision.message(), "TestKernel output logical elements must not overlap");

    const Status undecidable = ValidateInjectiveLayout(
            "TestKernel", LayoutInjectivity::kUnknown, "output");
    EXPECT_EQ(undecidable.code(), StatusCode::kUnimplemented);
    EXPECT_EQ(undecidable.message(),
              "TestKernel cannot prove output maps distinct coordinates to distinct offsets");
}

TEST(CpuAliasUtils, BuildStridedAddressFootprintRejectsInvalidGeometry) {
    std::array<float, 4> storage{};
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t strides[2] = {2, 1};
    constexpr int64_t negative_strides[2] = {-2, 1};
    constexpr int64_t single_axis_shape[1] = {2};

    const auto negative = BuildStridedAddressFootprint(
            storage.data(), shape, negative_strides, sizeof(float), "negative stride");
    const auto mismatched = BuildStridedAddressFootprint(
            storage.data(), shape, single_axis_shape, sizeof(float), "mismatched rank");
    const auto null_data = BuildStridedAddressFootprint(
            nullptr, shape, strides, sizeof(float), "null data");

    ASSERT_FALSE(negative.ok());
    EXPECT_EQ(negative.status().message(),
              "negative stride has invalid strided address geometry");
    ASSERT_FALSE(mismatched.ok());
    EXPECT_EQ(mismatched.status().message(),
              "mismatched rank has invalid strided address geometry");
    ASSERT_FALSE(null_data.ok());
    EXPECT_EQ(null_data.status().message(),
              "null data has invalid strided address geometry");
}

TEST(CpuAliasUtils, BuildStridedAddressFootprintRejectsOffsetOverflow) {
    std::array<float, 4> storage{};
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t huge_strides[2] = {std::numeric_limits<int64_t>::max(), 1};

    const auto footprint = BuildStridedAddressFootprint(
            storage.data(), shape, huge_strides, sizeof(float), "huge stride");

    ASSERT_FALSE(footprint.ok());
    EXPECT_EQ(footprint.status().message(), "huge stride address range overflow");
}

TEST(CpuAliasUtils, BuildStridedAddressFootprintRejectsAddressOverflow) {
    const auto near_address_limit = reinterpret_cast<const void*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    constexpr int64_t shape[1] = {2};
    constexpr int64_t strides[1] = {1};

    const auto footprint = BuildStridedAddressFootprint(
            near_address_limit, shape, strides, sizeof(float), "test footprint");

    ASSERT_FALSE(footprint.ok());
    EXPECT_EQ(footprint.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(footprint.status().message(), "test footprint address range overflow");
}

} // namespace
