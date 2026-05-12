#include <gtest/gtest.h>

#include "amstring/generic_layout_policy.hpp"
#include "amstring/layout_policy.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace {

using namespace aethermind;

template<typename Policy>
void WriteProbe(typename Policy::Storage& storage, typename Policy::ProbeWordType probe) noexcept {
    std::memcpy(storage.raw + Policy::kProbeByteOffset, &probe, sizeof(probe));
}

template<typename CharT>
class GenericLayoutPolicyTest : public ::testing::Test {};

using AmstringCharTypes = ::testing::Types<char, char8_t, char16_t, char32_t, wchar_t>;
TYPED_TEST_SUITE(GenericLayoutPolicyTest, AmstringCharTypes);

TYPED_TEST(GenericLayoutPolicyTest, CharTMatrixIsAvailable) {
    using CharT = TypeParam;

    static_assert(std::is_trivial_v<CharT>);
    static_assert(std::is_standard_layout_v<CharT>);

    SUCCEED() << "M0 skeleton: GenericLayoutPolicy tests are parameterized by CharT.";
}

TYPED_TEST(GenericLayoutPolicyTest, ExposesExpectedStorageConstants) {
    using Policy = GenericLayoutPolicy<TypeParam>;

    static_assert(AmStringLayoutPolicy<Policy, TypeParam>);
    static_assert(sizeof(typename Policy::Storage) == sizeof(typename Policy::ExternalRep));
    static_assert(Policy::kStorageBytes == sizeof(typename Policy::ExternalRep));
    static_assert(Policy::kStorageBytes == 24);
    static_assert(Policy::kSmallSlots == Policy::kStorageBytes / sizeof(TypeParam));
    static_assert(Policy::kSmallCapacity == Policy::kSmallSlots - 1);
    static_assert(Policy::kProbeBits == sizeof(TypeParam) * 8);
    static_assert(Policy::kSizeTypeBits == sizeof(std::size_t) * 8);
    static_assert(Policy::kPayloadBits == Policy::kSizeTypeBits - Policy::kProbeBits);
    static_assert(Policy::kProbeByteOffset == Policy::kStorageBytes - sizeof(TypeParam));
    static_assert(Policy::kExternalTag == Policy::kSmallCapacity + 1);
}

TYPED_TEST(GenericLayoutPolicyTest, InitEmptyCreatesValidSmallString) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;

    Policy::InitEmpty(storage);

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_FALSE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kSmall);
    EXPECT_EQ(Policy::size(storage), 0U);
    EXPECT_EQ(Policy::capacity(storage), Policy::kSmallCapacity);
    EXPECT_NE(Policy::data(storage), nullptr);
    EXPECT_EQ(Policy::data(storage)[0], CharT{});
    Policy::CheckInvariants(storage);
}

TYPED_TEST(GenericLayoutPolicyTest, InitSmallCopiesDataAndEncodesSize) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    constexpr std::size_t kSize = Policy::kSmallCapacity > 2 ? 2 : 1;
    const std::array<CharT, 2> source{static_cast<CharT>(1), static_cast<CharT>(2)};

    Policy::InitSmall(storage, source.data(), kSize);

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kSmall);
    EXPECT_EQ(Policy::size(storage), kSize);
    EXPECT_EQ(Policy::capacity(storage), Policy::kSmallCapacity);
    EXPECT_EQ(Policy::data(storage)[0], source[0]);
    EXPECT_EQ(Policy::data(storage)[kSize], CharT{});
    Policy::CheckInvariants(storage);
}

TYPED_TEST(GenericLayoutPolicyTest, DecodeProbeReturnsSmallState) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    constexpr std::size_t kSize = Policy::kSmallCapacity > 2 ? 2 : 1;
    const std::array<CharT, 2> source{static_cast<CharT>(1), static_cast<CharT>(2)};

    Policy::InitSmall(storage, source.data(), kSize);

    const auto decoded = Policy::DecodeProbe(storage);
    EXPECT_EQ(decoded.category, Policy::Category::kSmall);
    EXPECT_EQ(decoded.size, kSize);
    EXPECT_EQ(decoded.capacity, Policy::kSmallCapacity);
    EXPECT_TRUE(Policy::is_small(decoded));
    EXPECT_FALSE(Policy::is_external(decoded));
    EXPECT_EQ(Policy::size(decoded), kSize);
    EXPECT_EQ(Policy::capacity(decoded), Policy::kSmallCapacity);
    EXPECT_EQ(Policy::data(storage, decoded), storage.small);

    EXPECT_TRUE(Policy::TryPushBackInplace(storage, static_cast<CharT>(3), decoded));
    EXPECT_EQ(Policy::size(storage), kSize + 1);
    EXPECT_EQ(Policy::data(storage)[kSize], static_cast<CharT>(3));
    EXPECT_EQ(Policy::data(storage)[kSize + 1], CharT{});
}

TYPED_TEST(GenericLayoutPolicyTest, SetSmallSizeMaintainsTerminatorAndProbeMeta) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    std::array<CharT, Policy::kSmallCapacity == 0 ? 1 : Policy::kSmallCapacity> source{};
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<CharT>(i + 1);
    }

    Policy::InitSmall(storage, source.data(), source.size());
    Policy::SetSmallSize(storage, 0);

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_EQ(Policy::size(storage), 0U);
    EXPECT_EQ(Policy::data(storage)[0], CharT{});
    Policy::CheckInvariants(storage);
}

TYPED_TEST(GenericLayoutPolicyTest, InitExternalPacksTagAndCapacity) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    std::array<CharT, 9> buffer{};
    constexpr std::size_t kSize = 3;
    constexpr std::size_t kCapacity = 8;

    buffer[0] = static_cast<CharT>(11);
    buffer[1] = static_cast<CharT>(12);
    buffer[2] = static_cast<CharT>(13);
    Policy::InitExternal(storage, buffer.data(), kSize, kCapacity);

    EXPECT_FALSE(Policy::is_small(storage));
    EXPECT_TRUE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kExternal);
    EXPECT_EQ(Policy::size(storage), kSize);
    EXPECT_EQ(Policy::capacity(storage), kCapacity);
    EXPECT_EQ(Policy::data(storage), buffer.data());
    EXPECT_EQ(buffer[kSize], CharT{});
    Policy::CheckInvariants(storage);
}

TYPED_TEST(GenericLayoutPolicyTest, DecodeProbeReturnsExternalState) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    std::array<CharT, 9> buffer{};
    constexpr std::size_t kSize = 3;
    constexpr std::size_t kCapacity = 8;

    Policy::InitExternal(storage, buffer.data(), kSize, kCapacity);

    const auto decoded = Policy::DecodeProbe(storage);
    EXPECT_EQ(decoded.category, Policy::Category::kExternal);
    EXPECT_EQ(decoded.size, kSize);
    EXPECT_EQ(decoded.capacity, kCapacity);
    EXPECT_FALSE(Policy::is_small(decoded));
    EXPECT_TRUE(Policy::is_external(decoded));
    EXPECT_EQ(Policy::size(decoded), kSize);
    EXPECT_EQ(Policy::capacity(decoded), kCapacity);
    EXPECT_EQ(Policy::data(storage, decoded), buffer.data());

    EXPECT_TRUE(Policy::TryPushBackInplace(storage, static_cast<CharT>(14), decoded));
    EXPECT_EQ(Policy::size(storage), kSize + 1);
    EXPECT_EQ(buffer[kSize], static_cast<CharT>(14));
    EXPECT_EQ(buffer[kSize + 1], CharT{});
}

TYPED_TEST(GenericLayoutPolicyTest, SetExternalSizeAndCapacityMaintainExternalState) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    std::array<CharT, 17> buffer{};

    Policy::InitExternal(storage, buffer.data(), 2, 8);
    Policy::SetExternalSize(storage, 5);
    Policy::SetExternalCapacity(storage, 16);

    EXPECT_TRUE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kExternal);
    EXPECT_EQ(Policy::size(storage), 5U);
    EXPECT_EQ(Policy::capacity(storage), 16U);
    EXPECT_EQ(buffer[5], CharT{});
    Policy::CheckInvariants(storage);
}

TYPED_TEST(GenericLayoutPolicyTest, MaxExternalCapacityMatchesPayloadBits) {
    using Policy = GenericLayoutPolicy<TypeParam>;

    if constexpr (Policy::kPayloadBits == 0) {
        EXPECT_EQ(Policy::max_external_capacity(), 0U);
    } else {
        EXPECT_EQ(Policy::max_external_capacity(), (std::size_t{1} << Policy::kPayloadBits) - 1);
    }
}

TYPED_TEST(GenericLayoutPolicyTest, InvalidProbeMetaProducesInvalidCategory) {
    using Policy = GenericLayoutPolicy<TypeParam>;
    typename Policy::Storage storage;

    Policy::InitEmpty(storage);
    WriteProbe<Policy>(storage, static_cast<typename Policy::ProbeWordType>(Policy::kExternalTag + 1));

    EXPECT_FALSE(Policy::is_small(storage));
    EXPECT_FALSE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kInvalid);

    const auto decoded = Policy::DecodeProbe(storage);
    EXPECT_EQ(decoded.category, Policy::Category::kInvalid);
    EXPECT_EQ(decoded.size, 0U);
    EXPECT_EQ(decoded.capacity, 0U);
    EXPECT_FALSE(Policy::is_small(decoded));
    EXPECT_FALSE(Policy::is_external(decoded));
    EXPECT_EQ(Policy::size(decoded), 0U);
    EXPECT_EQ(Policy::capacity(decoded), 0U);
    EXPECT_EQ(Policy::data(storage, decoded), nullptr);
    EXPECT_FALSE(Policy::TryPushBackInplace(storage, static_cast<TypeParam>(1), decoded));
}

TYPED_TEST(GenericLayoutPolicyTest, CheckInvariantsRejectsInvalidCategory) {
    using Policy = GenericLayoutPolicy<TypeParam>;
    typename Policy::Storage storage;

    Policy::InitEmpty(storage);
    WriteProbe<Policy>(storage, static_cast<typename Policy::ProbeWordType>(Policy::kExternalTag + 1));

#ifndef NDEBUG
    EXPECT_DEATH(Policy::CheckInvariants(storage), "Check failed");
#endif
}

TYPED_TEST(GenericLayoutPolicyTest, CheckInvariantsRejectsExternalSizeBeyondCapacity) {
    using CharT = TypeParam;
    using Policy = GenericLayoutPolicy<CharT>;
    typename Policy::Storage storage;
    std::array<CharT, 9> buffer{};

    Policy::InitExternal(storage, buffer.data(), 2, 8);
    storage.external.size = 9;

#ifndef NDEBUG
    EXPECT_DEATH(Policy::CheckInvariants(storage), "Check failed");
#endif
}

}// namespace
