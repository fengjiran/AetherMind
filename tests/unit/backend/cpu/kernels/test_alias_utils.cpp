#include "aethermind/backend/cpu/kernels/common/alias_utils.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using namespace aethermind::cpu::detail;

static_assert(!std::is_aggregate_v<RowwiseAddressFootprint>);
static_assert(!std::is_default_constructible_v<RowwiseAddressFootprint>);
static_assert(std::is_nothrow_move_assignable_v<RowwiseAddressFootprint>);
static_assert(!std::is_aggregate_v<RowwiseViewAnalysis>);
static_assert(!std::is_default_constructible_v<RowwiseViewAnalysis>);
static_assert(std::is_nothrow_move_assignable_v<RowwiseViewAnalysis>);
static_assert(!std::is_aggregate_v<StridedAddressFootprint>);
static_assert(!std::is_default_constructible_v<StridedAddressFootprint>);
static_assert(std::is_nothrow_move_assignable_v<StridedAddressFootprint>);
static_assert(std::is_same_v<decltype(std::declval<StridedAddressFootprint>().envelope()),
                             ByteAddressRange>);

StatusOr<RowwiseViewAnalysis> AnalyzeRowwiseTestView(const void* data,
                                                     int64_t row_count,
                                                     int64_t column_count,
                                                     int64_t row_stride,
                                                     int64_t column_stride,
                                                     size_t element_size,
                                                     std::string_view context) {
    if (element_size != sizeof(float)) {
        return Status::InvalidArgument("test helper only supports float elements");
    }
    const std::array<int64_t, 2> shape = {row_count, column_count};
    const std::array<int64_t, 2> strides = {row_stride, column_stride};
    return AnalyzeRowwiseView(
            TensorView{data, DataType::Float32(), shape, strides}, context);
}

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

TEST(CpuAliasUtils, ViewFootprintOverloadsValidateViewsAndBindStridesCoherently) {
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t padded_strides[2] = {4, 1};
    std::array<float, 7> storage{};
    const TensorView immutable{storage.data(), DataType::Float32(), shape, padded_strides};
    const MutableTensorView mutable_view{
            storage.data(), DataType::Float32(), shape, padded_strides};

    const auto immutable_footprint = BuildStridedAddressFootprint(immutable, "immutable");
    const auto mutable_footprint = BuildStridedAddressFootprint(mutable_view, "mutable");
    const auto default_immutable = BuildStridedAddressFootprint(TensorView{}, "default immutable");
    const auto default_mutable =
            BuildStridedAddressFootprint(MutableTensorView{}, "default mutable");

    ASSERT_TRUE(immutable_footprint.ok()) << immutable_footprint.status().ToString();
    ASSERT_TRUE(mutable_footprint.ok()) << mutable_footprint.status().ToString();
    EXPECT_EQ(immutable_footprint->logical_element_count(), 6);
    EXPECT_EQ(immutable_footprint->max_element_offset(), 6);
    EXPECT_EQ(immutable_footprint->envelope().begin, mutable_footprint->envelope().begin);
    EXPECT_EQ(immutable_footprint->envelope().end, mutable_footprint->envelope().end);
    ASSERT_FALSE(default_immutable.ok());
    EXPECT_EQ(default_immutable.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_immutable.status().message(),
              "default immutable requires a valid tensor view");
    ASSERT_FALSE(default_mutable.ok());
    EXPECT_EQ(default_mutable.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_mutable.status().message(), "default mutable requires a valid tensor view");
}

TEST(CpuAliasUtils, ContiguousViewOverloadsHandleNormalRankZeroAndEmptyViews) {
    constexpr int64_t normal_shape[2] = {2, 3};
    constexpr int64_t normal_strides[2] = {3, 1};
    constexpr int64_t empty_shape[2] = {0, 3};
    constexpr int64_t empty_strides[2] = {3, 1};
    std::array<float, 6> normal_storage{};
    float scalar = 0.0F;

    const auto normal = BuildContiguousByteRange(
            TensorView{normal_storage.data(), DataType::Float32(), normal_shape, normal_strides},
            "normal");
    const auto mutable_normal = BuildContiguousByteRange(
            MutableTensorView{
                    normal_storage.data(), DataType::Float32(), normal_shape, normal_strides},
            "mutable normal");
    const auto rank_zero = BuildContiguousByteRange(
            TensorView{&scalar, DataType::Float32(), {}, {}}, "rank zero");
    const auto empty = BuildContiguousByteRange(
            TensorView{nullptr, DataType::Float32(), empty_shape, empty_strides}, "empty");
    const auto default_view = BuildContiguousByteRange(TensorView{}, "default contiguous");

    ASSERT_TRUE(normal.ok()) << normal.status().ToString();
    ASSERT_TRUE(mutable_normal.ok()) << mutable_normal.status().ToString();
    ASSERT_TRUE(rank_zero.ok()) << rank_zero.status().ToString();
    ASSERT_TRUE(empty.ok()) << empty.status().ToString();
    EXPECT_EQ(normal->end - normal->begin, 6 * sizeof(float));
    EXPECT_EQ(normal->begin, mutable_normal->begin);
    EXPECT_EQ(normal->end, mutable_normal->end);
    EXPECT_EQ(rank_zero->end - rank_zero->begin, sizeof(float));
    EXPECT_EQ(empty->begin, empty->end);
    ASSERT_FALSE(default_view.ok());
    EXPECT_EQ(default_view.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_view.status().message(),
              "default contiguous requires a valid tensor view");
}

TEST(CpuAliasUtils, ContiguousViewOverloadReportsEmptyViewStrideProductOverflow) {
    constexpr int64_t shape[3] = {0, std::numeric_limits<int64_t>::max(), 2};
    constexpr int64_t strides[3] = {0, 2, 1};
    const TensorView immutable{nullptr, DataType::Float32(), shape, strides};
    const MutableTensorView mutable_view{nullptr, DataType::Float32(), shape, strides};

    const auto immutable_footprint = BuildStridedAddressFootprint(immutable, "empty immutable");
    const auto immutable_range = BuildContiguousByteRange(immutable, "empty immutable");
    const auto mutable_range = BuildContiguousByteRange(mutable_view, "empty mutable");

    ASSERT_TRUE(immutable_footprint.ok()) << immutable_footprint.status().ToString();
    EXPECT_TRUE(immutable_footprint->is_empty());
    ASSERT_FALSE(immutable_range.ok());
    EXPECT_EQ(immutable_range.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(immutable_range.status().message(),
              "empty immutable contiguous stride product overflow");
    ASSERT_FALSE(mutable_range.ok());
    EXPECT_EQ(mutable_range.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(mutable_range.status().message(),
              "empty mutable contiguous stride product overflow");
}

TEST(CpuAliasUtils, ContiguousViewOverloadRejectsHoledViewBeforeFalseDisjointAliasProof) {
    constexpr int64_t holed_shape[1] = {4};
    constexpr int64_t holed_strides[1] = {2};
    constexpr int64_t scalar_shape[1] = {1};
    constexpr int64_t scalar_strides[1] = {1};
    std::array<float, 8> storage{};
    const TensorView holed{storage.data(), DataType::Float32(), holed_shape, holed_strides};
    const TensorView last_logical_element{
            storage.data() + 6, DataType::Float32(), scalar_shape, scalar_strides};

    // The raw helper cannot see holed strides. Treating this view as four
    // consecutive elements would falsely prove it disjoint from element 3.
    const auto raw_assumption = BuildContiguousByteRange(
            storage.data(), 4, sizeof(float), "raw contiguous assumption");
    const auto last_element = BuildContiguousByteRange(last_logical_element, "last element");
    const auto checked_view = BuildContiguousByteRange(holed, "holed view");

    ASSERT_TRUE(raw_assumption.ok()) << raw_assumption.status().ToString();
    ASSERT_TRUE(last_element.ok()) << last_element.status().ToString();
    EXPECT_FALSE(ByteRangesOverlap(*raw_assumption, *last_element));
    ASSERT_FALSE(checked_view.ok());
    EXPECT_EQ(checked_view.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(checked_view.status().message(),
              "holed view requires a row-major contiguous tensor view");
}

TEST(CpuAliasUtils, RowwiseViewAnalyzersRejectInvalidViews) {
    // The raw-parts constructors abort on invalid state, so a default-constructed
    // view is the only invalid view that can reach an analyzer. Both row-wise
    // entry points must report it the same way the strided overload does.
    const auto default_row_view = AnalyzeRowwiseView(TensorView{}, "default row view");
    const auto default_mutable =
            AnalyzeRowwiseView(MutableTensorView{}, "default mutable row view");
    const auto default_vector =
            AnalyzeRowwiseColumnVector(TensorView{}, "default column vector");
    const auto default_mutable_vector =
            AnalyzeRowwiseColumnVector(MutableTensorView{}, "default mutable vector");

    ASSERT_FALSE(default_row_view.ok());
    EXPECT_EQ(default_row_view.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_row_view.status().message(),
              "default row view requires a valid tensor view");
    ASSERT_FALSE(default_mutable.ok());
    EXPECT_EQ(default_mutable.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_mutable.status().message(),
              "default mutable row view requires a valid tensor view");
    ASSERT_FALSE(default_vector.ok());
    EXPECT_EQ(default_vector.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_vector.status().message(),
              "default column vector requires a valid tensor view");
    ASSERT_FALSE(default_mutable_vector.ok());
    EXPECT_EQ(default_mutable_vector.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(default_mutable_vector.status().message(),
              "default mutable vector requires a valid tensor view");
}

TEST(CpuAliasUtils, RowwiseViewAnalyzersStillRejectUnsupportedRanks) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    std::array<float, 4> storage{};
    float scalar = 0.0F;

    // Both views are valid, so the rank requirements and not the validity gate
    // must reject them.
    const auto rank_zero = AnalyzeRowwiseView(
            TensorView{&scalar, DataType::Float32(), {}, {}}, "rank zero");
    const auto matrix_as_vector = AnalyzeRowwiseColumnVector(
            MutableTensorView{storage.data(), DataType::Float32(), matrix_shape, matrix_strides},
            "matrix as vector");

    ASSERT_FALSE(rank_zero.ok());
    EXPECT_EQ(rank_zero.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(rank_zero.status().message(), "rank zero requires rank >= 1");
    ASSERT_FALSE(matrix_as_vector.ok());
    EXPECT_EQ(matrix_as_vector.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(matrix_as_vector.status().message(), "matrix as vector requires rank-1 tensor");
}

TEST(CpuAliasUtils, RowwiseFootprintsCanProveDisjointSeparateStorage) {
    std::array<float, 4> lhs_storage{};
    std::array<float, 4> rhs_storage{};
    const auto lhs = AnalyzeRowwiseTestView(
            lhs_storage.data(), 2, 2, 2, 1, sizeof(float), "lhs");
    const auto rhs = AnalyzeRowwiseTestView(
            rhs_storage.data(), 2, 2, 2, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_TRUE(ValidateRowwiseDisjoint(
                        "TestKernel", lhs->footprint(), "output", rhs->footprint(), "input")
                        .ok());
}

TEST(CpuAliasUtils, RowwiseFootprintsProveActualOverlapWithoutHoles) {
    std::array<float, 8> storage{};
    const auto lhs = AnalyzeRowwiseTestView(
            storage.data(), 2, 2, 4, 1, sizeof(float), "lhs");
    const auto rhs = AnalyzeRowwiseTestView(
            storage.data() + 1, 2, 2, 4, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ValidateRowwiseDisjoint(
                      "TestKernel", lhs->footprint(), "output", rhs->footprint(), "input")
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CpuAliasUtils, RowwiseFootprintsProveSharedAllocationRowsAreDisjoint) {
    std::array<float, 8> storage{};
    const auto lhs = AnalyzeRowwiseTestView(
            storage.data(), 2, 2, 4, 1, sizeof(float), "lhs");
    const auto rhs = AnalyzeRowwiseTestView(
            storage.data() + 2, 2, 2, 4, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_TRUE(ValidateRowwiseDisjoint(
                        "TestKernel", lhs->footprint(), "output", rhs->footprint(), "input")
                        .ok());
}

TEST(CpuAliasUtils, RowwiseFootprintsReportUnknownForColumnStrideHoles) {
    std::array<float, 8> storage{};
    const auto lhs = AnalyzeRowwiseTestView(
            storage.data(), 1, 4, 8, 2, sizeof(float), "lhs");
    const auto rhs = AnalyzeRowwiseTestView(
            storage.data() + 1, 1, 4, 8, 2, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ValidateRowwiseDisjoint(
                      "TestKernel", lhs->footprint(), "output", rhs->footprint(), "input")
                      .code(),
              StatusCode::kUnimplemented);
}

TEST(CpuAliasUtils, RowwiseFootprintsProveOverlapAtSharedBaseAddress) {
    std::array<float, 16> storage{};
    // Different layouts at the same base address: column-stride holes on the
    // lhs rule out the dense-row proof, so only the shared first element can
    // prove overlap.
    const auto lhs = AnalyzeRowwiseTestView(
            storage.data(), 1, 4, 8, 2, sizeof(float), "lhs");
    const auto rhs = AnalyzeRowwiseTestView(
            storage.data(), 2, 3, 8, 1, sizeof(float), "rhs");

    ASSERT_TRUE(lhs.ok()) << lhs.status().ToString();
    ASSERT_TRUE(rhs.ok()) << rhs.status().ToString();
    EXPECT_EQ(ValidateRowwiseDisjoint(
                      "TestKernel", lhs->footprint(), "output", rhs->footprint(), "input")
                      .code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ValidateRowwiseDisjoint(
                      "TestKernel", rhs->footprint(), "output", lhs->footprint(), "input")
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CpuAliasUtils, RowwiseEmptyFootprintsAreProvablyDisjoint) {
    std::array<float, 8> storage{};
    const auto zero_rows = AnalyzeRowwiseTestView(
            nullptr, 0, 4, 0, 0, sizeof(float), "zero rows");
    const auto zero_columns = AnalyzeRowwiseTestView(
            storage.data(), 2, 0, 0, 0, sizeof(float), "zero columns");
    const auto full = AnalyzeRowwiseTestView(
            storage.data(), 2, 2, 2, 1, sizeof(float), "full");

    ASSERT_TRUE(zero_rows.ok()) << zero_rows.status().ToString();
    ASSERT_TRUE(zero_columns.ok()) << zero_columns.status().ToString();
    ASSERT_TRUE(full.ok()) << full.status().ToString();
    EXPECT_TRUE(zero_rows->footprint().is_empty());
    EXPECT_TRUE(zero_columns->footprint().is_empty());
    EXPECT_FALSE(full->footprint().is_empty());
    EXPECT_EQ(zero_rows->footprint().envelope().begin, zero_rows->footprint().envelope().end);
    EXPECT_EQ(zero_columns->footprint().envelope().begin,
              zero_columns->footprint().envelope().end);
    EXPECT_TRUE(ValidateRowwiseOutputLayout("TestKernel", *zero_rows).ok());
    EXPECT_TRUE(ValidateRowwiseOutputLayout("TestKernel", *zero_columns).ok());
    EXPECT_TRUE(ValidateRowwiseDisjoint(
                        "TestKernel", zero_rows->footprint(), "output", full->footprint(), "input")
                        .ok());
    EXPECT_TRUE(
            ValidateRowwiseDisjoint(
                    "TestKernel", full->footprint(), "output", zero_columns->footprint(), "input")
                    .ok());
}

TEST(CpuAliasUtils, AnalyzeRowwiseViewRejectsAddressOverflow) {
    const auto near_address_limit = reinterpret_cast<const void*>(
            std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{3});
    const auto analysis = AnalyzeRowwiseTestView(
            near_address_limit, 1, 2, 2, 1, sizeof(float), "test layout");

    ASSERT_FALSE(analysis.ok());
    EXPECT_EQ(analysis.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuAliasUtils, AnalyzeRowwiseViewDistinguishesElementAndByteOverflow) {
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t element_overflow_strides[2] = {
            std::numeric_limits<int64_t>::max(), 1};
    constexpr int64_t byte_overflow_shape[2] = {2, 1};
    constexpr int64_t byte_overflow_strides[2] = {
            std::numeric_limits<int64_t>::max(), 1};
    std::array<float, 2> storage{};

    const auto element_overflow = AnalyzeRowwiseView(
            TensorView{storage.data(), DataType::Float32(), shape, element_overflow_strides},
            "element overflow");
    ASSERT_FALSE(element_overflow.ok());
    EXPECT_EQ(element_overflow.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(element_overflow.status().message(), "element overflow element offset overflow");

    const auto byte_overflow = AnalyzeRowwiseView(
            TensorView{storage.data(), DataType::Float32(), byte_overflow_shape, byte_overflow_strides},
            "byte overflow");
    ASSERT_FALSE(byte_overflow.ok());
    EXPECT_EQ(byte_overflow.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(byte_overflow.status().message(), "byte overflow address range overflow");
}

TEST(CpuAliasUtils, AnalyzeRowwiseViewSeparatesInputFactsFromOutputPolicy) {
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t overlapping_strides[2] = {1, 1};
    std::array<float, 6> storage{};

    const TensorView input{storage.data(), DataType::Float32(), shape, overlapping_strides};
    const MutableTensorView output{storage.data(), DataType::Float32(), shape, overlapping_strides};
    const auto input_analysis = AnalyzeRowwiseView(input, "read input");
    const auto output_analysis = AnalyzeRowwiseView(output, "output");

    ASSERT_TRUE(input_analysis.ok()) << input_analysis.status().ToString();
    ASSERT_TRUE(output_analysis.ok()) << output_analysis.status().ToString();
    EXPECT_FALSE(input_analysis->has_disjoint_row_envelopes());
    EXPECT_EQ(input_analysis->injectivity(), InjectivityClassification::kProvenNonInjective);

    const Status output_status = ValidateRowwiseOutputLayout("TestKernel", *output_analysis);
    EXPECT_EQ(output_status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(output_status.message(), "TestKernel requires disjoint output row envelopes");
}

TEST(CpuAliasUtils, AnalyzeRowwiseViewValidatesActualStridesButAllowsDerivedRankOneRowStride) {
    constexpr int64_t rank_one_shape[1] = {2};
    constexpr int64_t unit_stride[1] = {1};
    constexpr int64_t zero_stride[1] = {0};
    std::array<float, 2> storage{};

    const auto rank_one = AnalyzeRowwiseView(
            TensorView{storage.data(), DataType::Float32(), rank_one_shape, unit_stride}, "rank one");
    ASSERT_TRUE(rank_one.ok()) << rank_one.status().ToString();
    EXPECT_EQ(rank_one->row_count(), 1);
    EXPECT_EQ(rank_one->column_count(), 2);
    EXPECT_EQ(rank_one->row_stride(), 0);
    EXPECT_EQ(rank_one->column_stride(), 1);

    const auto zero_column_stride = AnalyzeRowwiseView(
            TensorView{storage.data(), DataType::Float32(), rank_one_shape, zero_stride}, "zero stride");
    ASSERT_FALSE(zero_column_stride.ok());
    EXPECT_EQ(zero_column_stride.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(zero_column_stride.status().message(), "zero stride requires positive strides");
}

TEST(CpuAliasUtils, AnalyzeRowwiseColumnVectorPreservesRankOneStrideGaps) {
    constexpr int64_t shape[1] = {2};
    constexpr int64_t stride[1] = {2};
    std::array<int64_t, 4> storage{};
    const TensorView vector{storage.data(), DataType::Int(64), shape, stride};

    const auto row_view = AnalyzeRowwiseView(vector, "row view");
    const auto column_view = AnalyzeRowwiseColumnVector(vector, "column view");

    ASSERT_TRUE(row_view.ok()) << row_view.status().ToString();
    ASSERT_TRUE(column_view.ok()) << column_view.status().ToString();
    EXPECT_EQ(row_view->row_count(), 1);
    EXPECT_EQ(row_view->column_count(), 2);
    EXPECT_EQ(row_view->row_stride(), 0);
    EXPECT_EQ(column_view->row_count(), 2);
    EXPECT_EQ(column_view->column_count(), 1);
    EXPECT_EQ(column_view->row_stride(), 2);
    EXPECT_EQ(column_view->column_stride(), 1);
}

TEST(CpuAliasUtils, ValidateRowwiseDisjointAcceptsProvenDisjointFootprints) {
    std::array<float, 4> output_storage{};
    std::array<float, 4> input_storage{};
    const auto output = AnalyzeRowwiseTestView(
            output_storage.data(), 2, 2, 2, 1, sizeof(float), "output");
    const auto input = AnalyzeRowwiseTestView(
            input_storage.data(), 2, 2, 2, 1, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateRowwiseDisjoint(
            "TestKernel", output->footprint(), "output", input->footprint(), "input");
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CpuAliasUtils, ValidateRowwiseDisjointRejectsProvenOverlap) {
    std::array<float, 8> storage{};
    const auto output = AnalyzeRowwiseTestView(
            storage.data() + 1, 2, 2, 4, 1, sizeof(float), "output");
    const auto input = AnalyzeRowwiseTestView(
            storage.data(), 2, 2, 4, 1, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateRowwiseDisjoint(
            "TestKernel", output->footprint(), "output", input->footprint(), "input");
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "TestKernel output must not overlap input");
}

TEST(CpuAliasUtils, ValidateRowwiseDisjointReportsUnknownAsUnimplemented) {
    std::array<float, 8> storage{};
    const auto output = AnalyzeRowwiseTestView(
            storage.data() + 1, 1, 4, 8, 2, sizeof(float), "output");
    const auto input = AnalyzeRowwiseTestView(
            storage.data(), 1, 4, 8, 2, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateRowwiseDisjoint(
            "TestKernel", output->footprint(), "output", input->footprint(), "input");
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
    EXPECT_EQ(status.message(),
              "TestKernel cannot prove output is disjoint from input "
              "for the requested strided layouts");
}

TEST(CpuAliasUtils, StridedFootprintExposesInjectivityProofsWithoutClassifierAccess) {
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
    std::array<float, 8> storage{};

    const auto colliding = BuildStridedAddressFootprint(
            storage.data(), colliding_shape, colliding_strides, sizeof(float), "colliding");
    const auto broadcast = BuildStridedAddressFootprint(
            storage.data(), broadcast_shape, broadcast_strides, sizeof(float), "broadcast");
    const auto irregular = BuildStridedAddressFootprint(
            storage.data(), irregular_shape, irregular_strides, sizeof(float), "irregular");
    const auto row_major = BuildStridedAddressFootprint(
            storage.data(), irregular_shape, row_major_strides, sizeof(float), "row major");
    const auto transposed = BuildStridedAddressFootprint(
            storage.data(), irregular_shape, transposed_strides, sizeof(float), "transposed");
    const auto padded = BuildStridedAddressFootprint(
            storage.data(), irregular_shape, padded_strides, sizeof(float), "padded");
    const auto unit_axis = BuildStridedAddressFootprint(
            storage.data(), unit_axis_shape, unit_axis_strides, sizeof(float), "unit axis");

    ASSERT_TRUE(colliding.ok()) << colliding.status().ToString();
    ASSERT_TRUE(broadcast.ok()) << broadcast.status().ToString();
    ASSERT_TRUE(irregular.ok()) << irregular.status().ToString();
    ASSERT_TRUE(row_major.ok()) << row_major.status().ToString();
    ASSERT_TRUE(transposed.ok()) << transposed.status().ToString();
    ASSERT_TRUE(padded.ok()) << padded.status().ToString();
    ASSERT_TRUE(unit_axis.ok()) << unit_axis.status().ToString();

    // Offsets {0, 1, 1, 2} collide, and a repeated element does too.
    EXPECT_EQ(colliding->injectivity(),
              InjectivityClassification::kProvenNonInjective);
    EXPECT_EQ(broadcast->injectivity(),
              InjectivityClassification::kProvenNonInjective);
    // Offsets {0, 2, 4, 3, 5, 7} are distinct, but the cheap proof cannot see it.
    EXPECT_EQ(irregular->injectivity(),
              InjectivityClassification::kUnknown);
    EXPECT_EQ(row_major->injectivity(),
              InjectivityClassification::kProvenInjective);
    EXPECT_EQ(transposed->injectivity(),
              InjectivityClassification::kProvenInjective);
    EXPECT_EQ(padded->injectivity(),
              InjectivityClassification::kProvenInjective);
    // Extent-1 axes never affect injectivity, whatever their stride.
    EXPECT_EQ(unit_axis->injectivity(),
              InjectivityClassification::kProvenInjective);
    EXPECT_EQ(ValidateLayoutInjectivity("TestKernel", colliding->injectivity(), "output").code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ValidateLayoutInjectivity("TestKernel", irregular->injectivity(), "output").code(),
              StatusCode::kUnimplemented);
    EXPECT_TRUE(ValidateLayoutInjectivity("TestKernel", transposed->injectivity(), "output").ok());
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
    EXPECT_EQ(transposed->logical_element_count(), 6);
    EXPECT_EQ(transposed->max_element_offset(), 5);
    EXPECT_TRUE(transposed->is_dense());
    EXPECT_FALSE(transposed->is_empty());
    EXPECT_EQ(padded->max_element_offset(), 6);
    EXPECT_FALSE(padded->is_dense());
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
    EXPECT_TRUE(empty->is_empty());
    EXPECT_EQ(empty->envelope().begin, empty->envelope().end);
    EXPECT_FALSE(scalar->is_empty());
    EXPECT_EQ(scalar->logical_element_count(), 1);
    EXPECT_EQ(scalar->max_element_offset(), 0);
    EXPECT_TRUE(scalar->is_dense());
    EXPECT_EQ(scalar->envelope().end - scalar->envelope().begin, sizeof(float));
}

TEST(CpuAliasUtils, ValidateStridedDisjointDistinguishesProofFromUnknown) {
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

    EXPECT_TRUE(ValidateStridedDisjoint(
                        "TestKernel", *dense_base, "output", *dense_far, "input")
                        .ok());
    EXPECT_EQ(ValidateStridedDisjoint(
                      "TestKernel", *dense_base, "output", *dense_shifted, "input")
                      .code(),
              StatusCode::kInvalidArgument);
    // A shared base address proves overlap even when neither view is dense.
    EXPECT_EQ(ValidateStridedDisjoint(
                      "TestKernel", *holed_base, "output", *dense_base, "input")
                      .code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ValidateStridedDisjoint(
                      "TestKernel", *holed_base, "output", *holed_shifted, "input")
                      .code(),
              StatusCode::kUnimplemented);
    EXPECT_EQ(ValidateStridedDisjoint(
                      "TestKernel", *holed_base, "output", *dense_shifted, "input")
                      .code(),
              StatusCode::kUnimplemented);
    // An empty view never overlaps, not even at a shared base address.
    EXPECT_TRUE(ValidateStridedDisjoint(
                        "TestKernel", *dense_base, "output", *empty_at_base, "input")
                        .ok());
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

TEST(CpuAliasUtils, ValidateContiguousDisjointAcceptsProvenDisjointRanges) {
    std::array<float, 4> output_storage{};
    std::array<float, 4> input_storage{};
    const auto output = BuildContiguousByteRange(
            output_storage.data(), 4, sizeof(float), "output");
    const auto input = BuildContiguousByteRange(
            input_storage.data(), 4, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateContiguousDisjoint(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CpuAliasUtils, ValidateContiguousDisjointRejectsProvenOverlap) {
    std::array<float, 4> storage{};
    const auto output = BuildContiguousByteRange(
            storage.data(), 4, sizeof(float), "output");
    const auto input = BuildContiguousByteRange(
            storage.data(), 4, sizeof(float), "input");

    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const Status status = ValidateContiguousDisjoint(
            "TestKernel", *output, "output", *input, "input");
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "TestKernel output must not overlap input");
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

} // namespace
