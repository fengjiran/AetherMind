#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <array>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using namespace aethermind::cpu::detail;

TEST(CpuAliasUtils, ByteRangesOverlapUsesHalfOpenIntervals) {
    EXPECT_TRUE(ByteRangesOverlap(ByteAddressRange{.begin = 100, .end = 110},
                                  ByteAddressRange{.begin = 105, .end = 115}));
    EXPECT_FALSE(ByteRangesOverlap(ByteAddressRange{.begin = 100, .end = 110},
                                   ByteAddressRange{.begin = 110, .end = 120}));
    EXPECT_FALSE(ByteRangesOverlap(ByteAddressRange{.begin = 100, .end = 100},
                                   ByteAddressRange{.begin = 90, .end = 110}));
}

TEST(CpuAliasUtils, ContiguousRangesSeparateAdjacentSlicesOfOneAllocation) {
    std::array<float, 8> storage{};
    const auto first_half = BuildContiguousByteRange(
            storage.data(), 4, sizeof(float), "first half");
    const auto second_half = BuildContiguousByteRange(
            storage.data() + 4, 4, sizeof(float), "second half");
    const auto shifted_tail = BuildContiguousByteRange(
            storage.data() + 3, 4, sizeof(float), "shifted tail");

    ASSERT_TRUE(first_half.ok()) << first_half.status().ToString();
    ASSERT_TRUE(second_half.ok()) << second_half.status().ToString();
    ASSERT_TRUE(shifted_tail.ok()) << shifted_tail.status().ToString();
    EXPECT_EQ(first_half->end - first_half->begin, 4 * sizeof(float));
    EXPECT_FALSE(ByteRangesOverlap(*first_half, *second_half));
    EXPECT_TRUE(ByteRangesOverlap(*first_half, *shifted_tail));
}

TEST(CpuAliasUtils, BuildContiguousByteRangeAcceptsEmptyViewWithoutData) {
    const auto empty = BuildContiguousByteRange(nullptr, 0, sizeof(float), "empty view");

    ASSERT_TRUE(empty.ok()) << empty.status().ToString();
    EXPECT_EQ(empty->begin, empty->end);
    EXPECT_FALSE(ByteRangesOverlap(*empty, ByteAddressRange{.begin = 0, .end = 64}));
}

TEST(CpuAliasUtils, BuildContiguousByteRangeRejectsInvalidGeometry) {
    std::array<float, 4> storage{};
    const auto null_data = BuildContiguousByteRange(nullptr, 4, sizeof(float), "null data");
    const auto zero_element_size = BuildContiguousByteRange(
            storage.data(), 4, 0, "zero element size");
    const auto negative_count = BuildContiguousByteRange(
            storage.data(), -1, sizeof(float), "negative count");

    ASSERT_FALSE(null_data.ok());
    EXPECT_EQ(null_data.status().message(),
              "null data has invalid contiguous address geometry");
    ASSERT_FALSE(zero_element_size.ok());
    EXPECT_EQ(zero_element_size.status().message(),
              "zero element size has invalid contiguous address geometry");
    ASSERT_FALSE(negative_count.ok());
    EXPECT_EQ(negative_count.status().message(),
              "negative count has invalid contiguous address geometry");
}

TEST(CpuAliasUtils, BuildContiguousByteRangeRejectsAddressOverflow) {
    const auto near_address_limit = reinterpret_cast<const void*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    const auto range = BuildContiguousByteRange(
            near_address_limit, 2, sizeof(float), "test range");

    ASSERT_FALSE(range.ok());
    EXPECT_EQ(range.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(range.status().message(), "test range address range overflow");
}

TEST(CpuAliasUtils, RowwiseFootprintsCanProveDisjointSeparateStorage) {
    std::array<float, 4> lhs_storage{};
    std::array<float, 4> rhs_storage{};
    const auto lhs = BuildRowwiseAddressFootprint(
            lhs_storage.data(), 2, 2, 2, 1, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressFootprint(
            rhs_storage.data(), 2, 2, 2, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseOverlap(*lhs, *rhs), OverlapClassification::kProvenDisjoint);
}

TEST(CpuAliasUtils, RowwiseFootprintsProveActualOverlapWithoutHoles) {
    std::array<float, 8> storage{};
    const auto lhs = BuildRowwiseAddressFootprint(
            storage.data(), 2, 2, 4, 1, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressFootprint(
            storage.data() + 1, 2, 2, 4, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseOverlap(*lhs, *rhs), OverlapClassification::kProvenOverlap);
}

TEST(CpuAliasUtils, RowwiseFootprintsProveSharedAllocationRowsAreDisjoint) {
    std::array<float, 8> storage{};
    const auto lhs = BuildRowwiseAddressFootprint(
            storage.data(), 2, 2, 4, 1, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressFootprint(
            storage.data() + 2, 2, 2, 4, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseOverlap(*lhs, *rhs), OverlapClassification::kProvenDisjoint);
}

TEST(CpuAliasUtils, RowwiseFootprintsReportUnknownForColumnStrideHoles) {
    std::array<float, 8> storage{};
    const auto lhs = BuildRowwiseAddressFootprint(
            storage.data(), 1, 4, 8, 2, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressFootprint(
            storage.data() + 1, 1, 4, 8, 2, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseOverlap(*lhs, *rhs), OverlapClassification::kUnknown);
}

TEST(CpuAliasUtils, RowwiseFootprintsProveOverlapAtSharedBaseAddress) {
    std::array<float, 16> storage{};
    // Different layouts at the same base address: column-stride holes on the
    // lhs rule out the dense-row proof, so only the shared first element can
    // prove overlap.
    const auto lhs = BuildRowwiseAddressFootprint(
            storage.data(), 1, 4, 8, 2, sizeof(float), "lhs");
    const auto rhs = BuildRowwiseAddressFootprint(
            storage.data(), 2, 3, 8, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ClassifyRowwiseOverlap(*lhs, *rhs), OverlapClassification::kProvenOverlap);
    EXPECT_EQ(ClassifyRowwiseOverlap(*rhs, *lhs), OverlapClassification::kProvenOverlap);
}

TEST(CpuAliasUtils, RowwiseEmptyFootprintsAreProvablyDisjoint) {
    std::array<float, 8> storage{};
    const auto zero_rows = BuildRowwiseAddressFootprint(
            nullptr, 0, 4, 4, 1, sizeof(float), "zero rows");
    const auto zero_columns = BuildRowwiseAddressFootprint(
            storage.data(), 2, 0, 4, 1, sizeof(float), "zero columns");
    const auto full = BuildRowwiseAddressFootprint(
            storage.data(), 2, 2, 2, 1, sizeof(float), "full");

    ASSERT_TRUE(zero_rows.ok()) << zero_rows.status().ToString();
    ASSERT_TRUE(zero_columns.ok()) << zero_columns.status().ToString();
    ASSERT_TRUE(full.ok()) << full.status().ToString();
    EXPECT_EQ(zero_rows->envelope.begin, zero_rows->envelope.end);
    EXPECT_EQ(zero_columns->envelope.begin, zero_columns->envelope.end);
    EXPECT_EQ(ClassifyRowwiseOverlap(*zero_rows, *full), OverlapClassification::kProvenDisjoint);
    EXPECT_EQ(ClassifyRowwiseOverlap(*zero_columns, *full),
              OverlapClassification::kProvenDisjoint);
    EXPECT_TRUE(ValidateRowwiseDisjoint("TestKernel", *zero_rows, "output", *full, "input").ok());
    EXPECT_TRUE(
            ValidateRowwiseDisjoint("TestKernel", *full, "output", *zero_columns, "input").ok());
}

TEST(CpuAliasUtils, BuildRowwiseAddressFootprintRejectsAddressOverflow) {
    const auto near_address_limit = reinterpret_cast<const void*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    const auto footprint = BuildRowwiseAddressFootprint(
            near_address_limit, 1, 2, 2, 1, sizeof(float), "test layout");

    ASSERT_FALSE(footprint.ok());
    EXPECT_EQ(footprint.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuAliasUtils, ValidateRowwiseDisjointAcceptsProvenDisjointFootprints) {
    std::array<float, 4> output_storage{};
    std::array<float, 4> input_storage{};
    const auto output = BuildRowwiseAddressFootprint(
            output_storage.data(), 2, 2, 2, 1, sizeof(float), "output");
    const auto input = BuildRowwiseAddressFootprint(
            input_storage.data(), 2, 2, 2, 1, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateRowwiseDisjoint(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CpuAliasUtils, ValidateRowwiseDisjointRejectsProvenOverlap) {
    std::array<float, 8> storage{};
    const auto output = BuildRowwiseAddressFootprint(
            storage.data() + 1, 2, 2, 4, 1, sizeof(float), "output");
    const auto input = BuildRowwiseAddressFootprint(
            storage.data(), 2, 2, 4, 1, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateRowwiseDisjoint(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "TestKernel output must not overlap input");
}

TEST(CpuAliasUtils, ValidateRowwiseDisjointReportsUnknownAsUnimplemented) {
    std::array<float, 8> storage{};
    const auto output = BuildRowwiseAddressFootprint(
            storage.data() + 1, 1, 4, 8, 2, sizeof(float), "output");
    const auto input = BuildRowwiseAddressFootprint(
            storage.data(), 1, 4, 8, 2, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateRowwiseDisjoint(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
    EXPECT_EQ(status.message(),
              "TestKernel cannot prove output is disjoint from input "
              "for the requested strided layouts");
}

TEST(CpuAliasUtils, ClassifyLayoutInjectivitySeparatesProofFromUnknown) {
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
              InjectivityClassification::kProvenNonInjective);
    EXPECT_EQ(ClassifyLayoutInjectivity(broadcast_shape, broadcast_strides),
              InjectivityClassification::kProvenNonInjective);
    // Offsets {0, 2, 4, 3, 5, 7} are distinct, but the cheap proof cannot see it.
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, irregular_strides),
              InjectivityClassification::kUnknown);
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, row_major_strides),
              InjectivityClassification::kProvenInjective);
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, transposed_strides),
              InjectivityClassification::kProvenInjective);
    EXPECT_EQ(ClassifyLayoutInjectivity(irregular_shape, padded_strides),
              InjectivityClassification::kProvenInjective);
    // Extent-1 axes never affect injectivity, whatever their stride.
    EXPECT_EQ(ClassifyLayoutInjectivity(unit_axis_shape, unit_axis_strides),
              InjectivityClassification::kProvenInjective);
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
    EXPECT_EQ(transposed->logical_element_count, 6);
    EXPECT_EQ(transposed->max_element_offset, 5);
    EXPECT_TRUE(transposed->is_dense);
    EXPECT_FALSE(transposed->is_empty);
    EXPECT_EQ(padded->max_element_offset, 6);
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
    EXPECT_EQ(scalar->logical_element_count, 1);
    EXPECT_EQ(scalar->max_element_offset, 0);
    EXPECT_TRUE(scalar->is_dense);
    EXPECT_EQ(scalar->envelope.end - scalar->envelope.begin, sizeof(float));
}

TEST(CpuAliasUtils, ClassifyStridedOverlapDistinguishesProofFromUnknown) {
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

    EXPECT_EQ(ClassifyStridedOverlap(*dense_base, *dense_far),
              OverlapClassification::kProvenDisjoint);
    EXPECT_EQ(ClassifyStridedOverlap(*dense_base, *dense_shifted),
              OverlapClassification::kProvenOverlap);
    // A shared base address proves overlap even when neither view is dense.
    EXPECT_EQ(ClassifyStridedOverlap(*holed_base, *dense_base),
              OverlapClassification::kProvenOverlap);
    EXPECT_EQ(ClassifyStridedOverlap(*holed_base, *holed_shifted),
              OverlapClassification::kUnknown);
    EXPECT_EQ(ClassifyStridedOverlap(*holed_base, *dense_shifted),
              OverlapClassification::kUnknown);
    // An empty view never overlaps, not even at a shared base address.
    EXPECT_EQ(ClassifyStridedOverlap(*dense_base, *empty_at_base),
              OverlapClassification::kProvenDisjoint);
}

TEST(CpuAliasUtils, ValidateLayoutInjectivityMapsProofOntoStatus) {
    const Status injective = ValidateLayoutInjectivity(
            "TestKernel", InjectivityClassification::kProvenInjective, "output");
    EXPECT_TRUE(injective.ok()) << injective.ToString();

    const Status collision = ValidateLayoutInjectivity(
            "TestKernel", InjectivityClassification::kProvenNonInjective, "output");
    EXPECT_EQ(collision.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(collision.message(), "TestKernel output logical elements must not overlap");

    const Status undecidable = ValidateLayoutInjectivity(
            "TestKernel", InjectivityClassification::kUnknown, "output");
    EXPECT_EQ(undecidable.code(), StatusCode::kUnimplemented);
    EXPECT_EQ(undecidable.message(),
              "TestKernel cannot prove output maps distinct coordinates to distinct offsets");
}

TEST(CpuAliasUtils, ValidateStridedDisjointMapsProofOntoStatus) {
    std::array<float, 8> storage{};
    constexpr int64_t dense_shape[2] = {2, 2};
    constexpr int64_t dense_strides[2] = {2, 1};
    constexpr int64_t holed_shape[1] = {4};
    constexpr int64_t holed_strides[1] = {2};

    const auto dense_base = BuildStridedAddressFootprint(
            storage.data(), dense_shape, dense_strides, sizeof(float), "dense base");
    const auto dense_shifted = BuildStridedAddressFootprint(
            storage.data() + 1, dense_shape, dense_strides, sizeof(float), "dense shifted");
    const auto dense_far = BuildStridedAddressFootprint(
            storage.data() + 7, dense_shape, dense_strides, sizeof(float), "dense far");
    const auto holed_shifted = BuildStridedAddressFootprint(
            storage.data() + 1, holed_shape, holed_strides, sizeof(float), "holed shifted");

    ASSERT_TRUE(dense_base.ok()) << dense_base.status().ToString();
    ASSERT_TRUE(dense_shifted.ok()) << dense_shifted.status().ToString();
    ASSERT_TRUE(dense_far.ok()) << dense_far.status().ToString();
    ASSERT_TRUE(holed_shifted.ok()) << holed_shifted.status().ToString();

    const Status disjoint = ValidateStridedDisjoint(
            "TestKernel", *dense_base, "output", *dense_far, "input");
    EXPECT_TRUE(disjoint.ok()) << disjoint.ToString();

    const Status overlap = ValidateStridedDisjoint(
            "TestKernel", *dense_base, "output", *dense_shifted, "input");
    EXPECT_EQ(overlap.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(overlap.message(), "TestKernel output must not overlap input");

    const Status unknown = ValidateStridedDisjoint(
            "TestKernel", *dense_base, "output", *holed_shifted, "input");
    EXPECT_EQ(unknown.code(), StatusCode::kUnimplemented);
    EXPECT_EQ(unknown.message(),
              "TestKernel cannot prove output is disjoint from input "
              "for the requested strided layouts");
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

TEST(CpuAliasUtils, HaveIdenticalViewMappingRejectsDifferentBasePointer) {
    std::array<float, 8> storage{};
    constexpr std::array<int64_t, 2> shape{2, 2};
    constexpr std::array<int64_t, 2> strides{2, 1};

    const TensorView input{storage.data(), DataType::Float32(), shape, strides};
    const MutableTensorView shifted{storage.data() + 4, DataType::Float32(), shape, strides};

    EXPECT_FALSE(HaveIdenticalViewMapping(input, shifted));
}

TEST(CpuAliasUtils, HaveIdenticalViewMappingRejectsDifferentDtype) {
    std::array<float, 8> storage{};
    constexpr std::array<int64_t, 2> shape{2, 2};
    constexpr std::array<int64_t, 2> strides{2, 1};

    const TensorView input{storage.data(), DataType::Float32(), shape, strides};
    // Same itemsize as Float32, so only the dtype identity can reject it.
    const MutableTensorView output{storage.data(), DataType::Int(32), shape, strides};

    EXPECT_FALSE(HaveIdenticalViewMapping(input, output));
}

TEST(CpuAliasUtils, HaveIdenticalViewMappingRejectsDifferentRank) {
    std::array<float, 8> storage{};
    constexpr std::array<int64_t, 2> shape{1, 4};
    constexpr std::array<int64_t, 2> strides{4, 1};
    constexpr std::array<int64_t, 1> flat_shape{4};
    constexpr std::array<int64_t, 1> flat_strides{1};

    const TensorView input{storage.data(), DataType::Float32(), shape, strides};
    const MutableTensorView output{storage.data(), DataType::Float32(), flat_shape, flat_strides};

    EXPECT_FALSE(HaveIdenticalViewMapping(input, output));
}

TEST(CpuAliasUtils, HaveIdenticalViewMappingRejectsSingleAxisDifference) {
    std::array<float, 8> storage{};
    constexpr std::array<int64_t, 2> shape{2, 2};
    constexpr std::array<int64_t, 2> strides{2, 1};
    constexpr std::array<int64_t, 2> transposed_strides{1, 2};
    constexpr std::array<int64_t, 2> wider_shape{2, 3};
    constexpr std::array<int64_t, 2> wider_strides{3, 1};

    const TensorView input{storage.data(), DataType::Float32(), shape, strides};
    const MutableTensorView transposed{storage.data(), DataType::Float32(), shape, transposed_strides};
    const MutableTensorView wider{storage.data(), DataType::Float32(), wider_shape, wider_strides};

    EXPECT_FALSE(HaveIdenticalViewMapping(input, transposed));
    EXPECT_FALSE(HaveIdenticalViewMapping(input, wider));
}

#ifndef NDEBUG

TEST(CpuAliasUtils, ClassifyLayoutInjectivityOverrankDeathTest) {
    // Debug builds assert the fixed-rank precondition instead of proceeding.
    constexpr std::array<int64_t, 9> shape{2, 2, 2, 2, 2, 2, 2, 2, 2};
    constexpr std::array<int64_t, 9> strides{256, 128, 64, 32, 16, 8, 4, 2, 1};

    EXPECT_DEATH(ClassifyLayoutInjectivity(shape, strides), "Check failed");
}

TEST(CpuAliasUtils, ClassifyLayoutInjectivityMismatchedSizeDeathTest) {
    constexpr std::array<int64_t, 2> shape{2, 2};
    constexpr std::array<int64_t, 1> strides{1};

    EXPECT_DEATH(ClassifyLayoutInjectivity(shape, strides), "Check failed");
}

#else

TEST(CpuAliasUtils, ClassifyLayoutInjectivityReportsViolatedPreconditionsAsUnknown) {
    // Release builds must not index past the fixed-rank axis buffer, so
    // violated preconditions are conservatively reported as undecidable.
    constexpr std::array<int64_t, 9> overrank_shape{2, 2, 2, 2, 2, 2, 2, 2, 2};
    constexpr std::array<int64_t, 9> overrank_strides{256, 128, 64, 32, 16, 8, 4, 2, 1};
    constexpr std::array<int64_t, 2> shape{2, 2};
    constexpr std::array<int64_t, 1> mismatched_strides{1};

    EXPECT_EQ(ClassifyLayoutInjectivity(overrank_shape, overrank_strides),
              InjectivityClassification::kUnknown);
    EXPECT_EQ(ClassifyLayoutInjectivity(shape, mismatched_strides),
              InjectivityClassification::kUnknown);
}

#endif

} // namespace
