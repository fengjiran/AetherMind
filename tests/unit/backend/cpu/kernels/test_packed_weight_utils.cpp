#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/cpu/kernels/common/packed_weight_utils.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace {

using namespace aethermind;

PackedWeightView MakeIdentityPackedWeight(const void* data,
                                          size_t nbytes,
                                          std::span<const int64_t> shape) {
    return PackedWeightView{
            .data = data,
            .nbytes = nbytes,
            .logical_dtype = DataType::Float32(),
            .logical_shape = shape,
            .recipe_layout = cpu::kCpuIdentityPackingLayout,
            .recipe_alignment = cpu::kCpuIdentityPackingAlignment,
            .alignment = cpu::kCpuIdentityPackingAlignment,
    };
}


PackedWeightView MakeBpanelPackedWeight(const void* data,
                                        size_t nbytes,
                                        std::span<const int64_t> shape) {
    return PackedWeightView{
            .data = data,
            .nbytes = nbytes,
            .logical_dtype = DataType::Float32(),
            .logical_shape = shape,
            .recipe_layout = cpu::kCpuBPanelF32V1Avx2Layout,
            .recipe_alignment = cpu::kCpuBPanelF32V1Alignment,
            .alignment = cpu::kCpuBPanelF32V1Alignment,
    };
}

TEST(CpuBpanelPackedWeight, RequiresExactLayoutSizeAndRealAlignment) {
    alignas(64) std::array<std::byte, 32768> storage{};
    constexpr std::array<int64_t, 2> shape{1, 1};
    constexpr size_t required = 32768;
    EXPECT_TRUE(cpu::detail::ValidateBPanelF32PackedWeight(
                        MakeBpanelPackedWeight(storage.data(), required, shape),
                        shape, "PackedWeightUtilsTest")
                        .ok());
    EXPECT_EQ(cpu::detail::ValidateBPanelF32PackedWeight(
                      MakeBpanelPackedWeight(storage.data(), required + 4, shape),
                      shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(cpu::detail::ValidateBPanelF32PackedWeight(
                      MakeBpanelPackedWeight(storage.data() + 1, required, shape),
                      shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CpuIdentityPackedWeight, ValidatesRankOneAndRankTwoShapes) {
    alignas(64) float rank_one_storage[3]{};
    constexpr std::array<int64_t, 1> rank_one_shape{3};
    EXPECT_TRUE(cpu::detail::ValidateIdentityPackedWeight(
                        MakeIdentityPackedWeight(rank_one_storage, sizeof(rank_one_storage),
                                                 rank_one_shape),
                        rank_one_shape, "PackedWeightUtilsTest")
                        .ok());

    alignas(64) float rank_two_storage[6]{};
    constexpr std::array<int64_t, 2> rank_two_shape{2, 3};
    EXPECT_TRUE(cpu::detail::ValidateIdentityPackedWeight(
                        MakeIdentityPackedWeight(rank_two_storage, sizeof(rank_two_storage),
                                                 rank_two_shape),
                        rank_two_shape, "PackedWeightUtilsTest")
                        .ok());
}

TEST(CpuIdentityPackedWeight, RejectsMalformedMetadataRecipeAndStorage) {
    alignas(64) float storage[3]{};
    constexpr std::array<int64_t, 1> expected_shape{3};
    constexpr std::array<int64_t, 1> wrong_shape{2};

    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      MakeIdentityPackedWeight(storage, sizeof(storage), wrong_shape),
                      expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 2> rank_mismatch_shape{1, 3};
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      MakeIdentityPackedWeight(storage, sizeof(storage), rank_mismatch_shape),
                      expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    PackedWeightView wrong_dtype =
            MakeIdentityPackedWeight(storage, sizeof(storage), expected_shape);
    wrong_dtype.logical_dtype = DataType::Float(16);
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      wrong_dtype, expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    PackedWeightView wrong_recipe =
            MakeIdentityPackedWeight(storage, sizeof(storage), expected_shape);
    wrong_recipe.recipe_layout = "different_recipe";
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      wrong_recipe, expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    wrong_recipe = MakeIdentityPackedWeight(storage, sizeof(storage), expected_shape);
    wrong_recipe.recipe_alignment = cpu::kCpuIdentityPackingAlignment / 2U;
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      wrong_recipe, expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    wrong_recipe = MakeIdentityPackedWeight(storage, sizeof(storage), expected_shape);
    wrong_recipe.alignment = cpu::kCpuIdentityPackingAlignment / 2U;
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      wrong_recipe, expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      MakeIdentityPackedWeight(storage, sizeof(float), expected_shape),
                      expected_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CpuIdentityPackedWeight, ChecksNegativeDimensionsAndProducts) {
    alignas(64) float storage[1]{};
    constexpr std::array<int64_t, 1> negative_shape{-1};
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      MakeIdentityPackedWeight(storage, sizeof(storage), negative_shape),
                      negative_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 2> element_overflow_shape{
            std::numeric_limits<int64_t>::max(), 2};
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      MakeIdentityPackedWeight(storage, sizeof(storage), element_overflow_shape),
                      element_overflow_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kOverflow);

    constexpr std::array<int64_t, 2> byte_overflow_shape{
            std::numeric_limits<int64_t>::max(), 1};
    EXPECT_EQ(cpu::detail::ValidateIdentityPackedWeight(
                      MakeIdentityPackedWeight(storage, sizeof(storage), byte_overflow_shape),
                      byte_overflow_shape, "PackedWeightUtilsTest")
                      .code(),
              StatusCode::kOverflow);
}

} // namespace
