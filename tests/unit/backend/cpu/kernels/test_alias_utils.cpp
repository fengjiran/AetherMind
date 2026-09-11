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

} // namespace
