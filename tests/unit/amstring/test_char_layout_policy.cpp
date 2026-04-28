#include <gtest/gtest.h>

#include "amstring/char_layout_policy.hpp"
#include "amstring/config.hpp"
#include "amstring/layout_policy.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using namespace aethermind;

using Policy = CharLayoutPolicy;

void WriteProbeByte(Policy::Storage& storage, std::uint8_t probe) noexcept {
    storage.raw[Policy::kProbeByteOffset] = static_cast<std::byte>(probe);
}

std::uint8_t EncodedSmallProbe(Policy::SizeType size) noexcept {
    const auto meta = Policy::EncodeSmallSizeToMeta(size);
    if constexpr (config::kIsLittleEndian) {
        return static_cast<std::uint8_t>(meta);
    } else {
        return static_cast<std::uint8_t>(meta << Policy::kCategoryBits);
    }
}

TEST(CharLayoutPolicyTest, ExposesExpectedStorageConstants) {
    static_assert(AmStringLayoutPolicy<Policy, char>);
    static_assert(sizeof(Policy::ExternalRep) == 24);
    static_assert(sizeof(Policy::Storage) == sizeof(Policy::ExternalRep));
    static_assert(sizeof(Policy::Storage) == 24);
    static_assert(std::is_trivially_copyable_v<Policy::Storage>);
    static_assert(Policy::kStorageBytes == sizeof(Policy::ExternalRep));
    static_assert(Policy::kSmallSlots == Policy::kStorageBytes);
    static_assert(Policy::kSmallCapacity == 23);
    static_assert(Policy::kCategoryBits == 2);
    static_assert(Policy::kSizeTypeBits == sizeof(std::size_t) * 8);
    static_assert(Policy::kPackedWordBits == Policy::kSizeTypeBits);
    static_assert(Policy::kPayloadBits == Policy::kSizeTypeBits - Policy::kCategoryBits);
    static_assert(Policy::kProbeByteOffset == Policy::kStorageBytes - 1);
    static_assert(Policy::kSmallMarker == 0x00U);

    EXPECT_EQ(Policy::kCategoryMask, config::kIsLittleEndian ? 0xC0U : 0x03U);
    EXPECT_EQ(Policy::kSmallMetaMask, config::kIsLittleEndian ? 0x3FU : 0xFCU);
    EXPECT_EQ(Policy::kExternalMarker, config::kIsLittleEndian ? 0x80U : 0x02U);
}

TEST(CharLayoutPolicyTest, InitEmptyCreatesValidSmallString) {
    Policy::Storage storage;

    Policy::InitEmpty(storage);

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_FALSE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kSmall);
    EXPECT_EQ(Policy::size(storage), 0U);
    EXPECT_EQ(Policy::capacity(storage), Policy::kSmallCapacity);
    EXPECT_EQ(Policy::data(storage), storage.small);
    EXPECT_EQ(Policy::data(storage)[0], char{});
    EXPECT_EQ(Policy::GetProbeByte(storage), EncodedSmallProbe(0));
    Policy::CheckInvariants(storage);
}

TEST(CharLayoutPolicyTest, InitSmallCopiesDataAndEncodesSize) {
    Policy::Storage storage;
    constexpr std::array<char, 5> kSource{'a', 'm', 's', 't', 'r'};

    Policy::InitSmall(storage, kSource.data(), kSource.size());

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kSmall);
    EXPECT_EQ(Policy::size(storage), kSource.size());
    EXPECT_EQ(Policy::capacity(storage), Policy::kSmallCapacity);
    EXPECT_EQ(Policy::data(storage), storage.small);
    EXPECT_EQ(Policy::data(storage)[0], kSource[0]);
    EXPECT_EQ(Policy::data(storage)[kSource.size()], char{});
    EXPECT_EQ(Policy::GetProbeByte(storage), EncodedSmallProbe(kSource.size()));
    Policy::CheckInvariants(storage);
}

TEST(CharLayoutPolicyTest, InitSmallMaxCapacityAliasesTerminatorAndProbe) {
    Policy::Storage storage;
    std::array<char, Policy::kSmallCapacity> source{};
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<char>('a' + (i % 26));
    }

    Policy::InitSmall(storage, source.data(), source.size());

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_EQ(Policy::size(storage), Policy::kSmallCapacity);
    EXPECT_EQ(Policy::data(storage)[Policy::kSmallCapacity], char{});
    EXPECT_EQ(Policy::GetProbeByte(storage), EncodedSmallProbe(Policy::kSmallCapacity));
    EXPECT_EQ(Policy::GetProbeByte(storage), 0U);
    Policy::CheckInvariants(storage);
}

TEST(CharLayoutPolicyTest, SetSmallSizeMaintainsTerminatorAndProbeMeta) {
    Policy::Storage storage;
    constexpr std::array<char, 6> kSource{'s', 't', 'r', 'i', 'n', 'g'};

    Policy::InitSmall(storage, kSource.data(), kSource.size());
    Policy::SetSmallSize(storage, 2);

    EXPECT_TRUE(Policy::is_small(storage));
    EXPECT_EQ(Policy::size(storage), 2U);
    EXPECT_EQ(Policy::data(storage)[2], char{});
    EXPECT_EQ(Policy::GetProbeByte(storage), EncodedSmallProbe(2));
    Policy::CheckInvariants(storage);
}

TEST(CharLayoutPolicyTest, InitExternalPacksMarkerAndCapacity) {
    Policy::Storage storage;
    std::array<char, 9> buffer{};
    constexpr std::size_t kSize = 3;
    constexpr std::size_t kCapacity = 8;

    buffer[0] = 'a';
    buffer[1] = 'm';
    buffer[2] = 's';
    Policy::InitExternal(storage, buffer.data(), kSize, kCapacity);

    EXPECT_FALSE(Policy::is_small(storage));
    EXPECT_TRUE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kExternal);
    EXPECT_EQ(Policy::size(storage), kSize);
    EXPECT_EQ(Policy::capacity(storage), kCapacity);
    EXPECT_EQ(Policy::data(storage), buffer.data());
    EXPECT_EQ(buffer[kSize], char{});
    EXPECT_EQ(Policy::UnpackCapacity(storage.external.capacity_with_tag), kCapacity);
    EXPECT_EQ(Policy::UnpackMarker(storage.external.capacity_with_tag), Policy::kExternalMarker);
    EXPECT_EQ(Policy::GetProbeByte(storage) & Policy::kCategoryMask, Policy::kExternalMarker);
    Policy::CheckInvariants(storage);
}

TEST(CharLayoutPolicyTest, SetExternalSizeAndCapacityMaintainExternalState) {
    Policy::Storage storage;
    std::array<char, 17> buffer{};

    Policy::InitExternal(storage, buffer.data(), 2, 8);
    Policy::SetExternalSize(storage, 5);
    Policy::SetExternalCapacity(storage, 16);

    EXPECT_TRUE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kExternal);
    EXPECT_EQ(Policy::size(storage), 5U);
    EXPECT_EQ(Policy::capacity(storage), 16U);
    EXPECT_EQ(buffer[5], char{});
    EXPECT_EQ(Policy::UnpackMarker(storage.external.capacity_with_tag), Policy::kExternalMarker);
    Policy::CheckInvariants(storage);
}

TEST(CharLayoutPolicyTest, SmallEncodingRoundTripsAllValidSizes) {
    for (std::size_t size = 0; size <= Policy::kSmallCapacity; ++size) {
        const auto probe = EncodedSmallProbe(size);

        EXPECT_EQ(Policy::DecodeSmallSizeFromMeta(probe), size);
        EXPECT_EQ(probe & Policy::kCategoryMask, Policy::kSmallMarker);
    }
}

TEST(CharLayoutPolicyTest, PackCapacityRoundTripsMarkerAndCapacity) {
    constexpr std::array<std::size_t, 4> kCapacities{0, 1, 32, 4096};

    for (const auto capacity : kCapacities) {
        const auto packed = Policy::PackCapacityWithTag(capacity);

        EXPECT_EQ(Policy::UnpackCapacity(packed), capacity);
        EXPECT_EQ(Policy::UnpackMarker(packed), Policy::kExternalMarker);
    }
}

TEST(CharLayoutPolicyTest, MaxExternalCapacityMatchesPayloadBits) {
    EXPECT_EQ(Policy::max_external_capacity(), (std::size_t{1} << Policy::kPayloadBits) - 1);
    EXPECT_GT(Policy::max_external_capacity(), (std::size_t{1} << (Policy::kSizeTypeBits - 8)) - 1);
}

TEST(CharLayoutPolicyTest, InvalidMarkerProducesInvalidCategory) {
    Policy::Storage storage;

    Policy::InitEmpty(storage);
    WriteProbeByte(storage, config::kIsLittleEndian ? 0x40U : 0x01U);

    EXPECT_FALSE(Policy::is_small(storage));
    EXPECT_FALSE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kInvalid);
}

TEST(CharLayoutPolicyTest, InvalidSmallMetaProducesInvalidCategory) {
    Policy::Storage storage;

    Policy::InitEmpty(storage);
    WriteProbeByte(storage, config::kIsLittleEndian ? 0x3FU : 0xFCU);

    EXPECT_FALSE(Policy::is_small(storage));
    EXPECT_FALSE(Policy::is_external(storage));
    EXPECT_EQ(Policy::category(storage), Policy::Category::kInvalid);
}

TEST(CharLayoutPolicyTest, CheckInvariantsRejectsInvalidCategory) {
    Policy::Storage storage;

    Policy::InitEmpty(storage);
    WriteProbeByte(storage, config::kIsLittleEndian ? 0x40U : 0x01U);

#ifndef NDEBUG
    EXPECT_DEATH(Policy::CheckInvariants(storage), "Check failed");
#endif
}

TEST(CharLayoutPolicyTest, CheckInvariantsRejectsExternalSizeBeyondCapacity) {
    Policy::Storage storage;
    std::array<char, 9> buffer{};

    Policy::InitExternal(storage, buffer.data(), 2, 8);
    storage.external.size = 9;

#ifndef NDEBUG
    EXPECT_DEATH(Policy::CheckInvariants(storage), "Check failed");
#endif
}

}// namespace
