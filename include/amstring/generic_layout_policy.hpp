// generic_layout_policy.hpp - Generic layout policy for multi-CharT amstring support
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_GENERIC_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_GENERIC_LAYOUT_POLICY_HPP

#include "config.hpp"
#include "invariant.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace aethermind {

template<std::size_t N>
struct UIntOfSize;

template<>
struct UIntOfSize<1> {
    using Type = std::uint8_t;
};

template<>
struct UIntOfSize<2> {
    using Type = std::uint16_t;
};

template<>
struct UIntOfSize<4> {
    using Type = std::uint32_t;
};

template<>
struct UIntOfSize<8> {
    using Type = std::uint64_t;
};

template<typename CharT>
using MetaWordT = UIntOfSize<sizeof(CharT)>::Type;

template<typename CharT>
struct GenericLayoutPolicy {
    static_assert(std::is_trivial_v<CharT>, "CharT must be trivial.");
    static_assert(std::is_standard_layout_v<CharT>, "CharT must be standard layout.");
    static_assert(sizeof(CharT) == 1 || sizeof(CharT) == 2 || sizeof(CharT) == 4 || sizeof(CharT) == 8, "Unsupported CharT width.");
    static_assert(sizeof(std::size_t) >= sizeof(CharT), "CharT cannot be wider than size_t.");

    using ValueType = CharT;
    using SizeType = std::size_t;
    using WordType = MetaWordT<CharT>;

    struct ExternalRep {
        CharT* data;
        SizeType size;
        SizeType capacity_with_tag;
    };

    union Storage {
        CharT small[sizeof(ExternalRep) / sizeof(CharT)];
        ExternalRep external;
        std::byte raw[sizeof(ExternalRep)]{};
    };

    using ExternalType = ExternalRep;
    using StorageType = Storage;

    enum class Category : std::uint8_t {
        Small,
        External,
        Invalid
    };

    static constexpr SizeType kStorageBytes = sizeof(ExternalRep);
    static constexpr SizeType kSmallSlots = kStorageBytes / sizeof(CharT);
    static constexpr SizeType kMetaSlot = kSmallSlots - 1;
    static constexpr SizeType kSmallCapacity = kSmallSlots - 1;
    static constexpr SizeType kProbeBits = sizeof(CharT) * 8;
    static constexpr SizeType kWordBits = sizeof(SizeType) * 8;
    static constexpr SizeType kPayloadBits = kWordBits - kProbeBits;
    static constexpr SizeType kProbeByteOffset = kStorageBytes - sizeof(CharT);
    static constexpr WordType kExternalTag = static_cast<WordType>(kSmallCapacity + 1);
    static constexpr WordType kMaxSmallMeta = static_cast<WordType>(kSmallCapacity);

    static_assert(sizeof(StorageType) == sizeof(ExternalType));

    static bool is_small(const StorageType& storage) noexcept {
        return ProbeMeta(storage) <= kMaxSmallMeta;
    }

    static bool is_external(const StorageType& storage) noexcept {
        return ProbeMeta(storage) == kExternalTag;
    }

    static Category category(const StorageType& storage) noexcept {
        const WordType meta = ProbeMeta(storage);
        if (meta <= kMaxSmallMeta) {
            return Category::Small;
        }
        if (meta == kExternalTag) {
            return Category::External;
        }
        return Category::Invalid;
    }

    static constexpr SizeType max_external_capacity() noexcept {
        if constexpr (kPayloadBits == 0) {
            return 0;
        } else if constexpr (kPayloadBits >= kWordBits) {
            return std::numeric_limits<SizeType>::max();
        } else {
            return (SizeType{1} << kPayloadBits) - 1;
        }
    }

    static const CharT* data(const StorageType& storage) noexcept {
        return is_small(storage) ? storage.small : storage.external.data;
    }

    static CharT* data(StorageType& storage) noexcept {
        return is_small(storage) ? storage.small : storage.external.data;
    }

    static SizeType size(const StorageType& storage) noexcept {
        if (is_small(storage)) {
            return DecodeSmallSizeFromMeta(ProbeMeta(storage));
        }
        return storage.external.size;
    }

    static SizeType capacity(const StorageType& storage) noexcept {
        if (is_small(storage)) {
            return kSmallCapacity;
        }
        return UnpackCapacity(storage.external.capacity_with_tag);
    }

    static void InitEmpty(StorageType& storage) noexcept {
        ClearStorage(storage);
        SetSmallSize(storage, 0);
    }

    static void InitSmall(StorageType& storage, const CharT* src, SizeType size) noexcept {
        assert(size <= kSmallCapacity);
        ClearStorage(storage);
        if (src != nullptr && size != 0) {
            std::memcpy(storage.small, src, size * sizeof(CharT));
        }
        SetSmallSize(storage, size);
    }

    static void InitExternal(StorageType& storage, CharT* ptr, SizeType size, SizeType capacity) noexcept {
        assert(ptr != nullptr);
        assert(size <= capacity);
        assert(capacity <= max_external_capacity());
        storage.external.data = ptr;
        storage.external.size = size;
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity, kExternalTag);
        ptr[size] = CharT{};
    }

    static void SetSmallSize(StorageType& storage, SizeType size) noexcept {
        assert(is_small(storage));
        assert(size <= kSmallCapacity);
        storage.small[size] = CharT{};
        SetProbeMeta(storage, EncodeSmallSizeToMeta(size));
    }

    static void SetExternalSize(StorageType& storage, SizeType size) noexcept {
        assert(is_external(storage));
        assert(size <= capacity(storage));
        storage.external.size = size;
        storage.external.data[size] = CharT{};
    }

    static void SetExternalCapacity(StorageType& storage, SizeType capacity) noexcept {
        assert(is_external(storage));
        assert(storage.external.size <= capacity);
        assert(capacity <= max_external_capacity());
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity, kExternalTag);
    }

    static void CheckInvariants(const StorageType& storage) noexcept {
        assert(category(storage) != Category::Invalid);
        check_invariant_impl(data(storage), size(storage), capacity(storage));
    }

    static WordType ProbeMeta(const StorageType& storage) noexcept {
        WordType meta{};
        std::memcpy(&meta, storage.raw + kProbeByteOffset, sizeof(WordType));
        return meta;
    }

    static void SetProbeMeta(StorageType& storage, WordType meta) noexcept {
        StoreProbeMetaAsCharT(storage, meta);
    }

    static void StoreProbeMetaAsCharT(StorageType& storage, WordType meta) noexcept {
        std::memcpy(storage.raw + kProbeByteOffset, &meta, sizeof(WordType));
    }

    static constexpr SizeType EncodeSmallSizeToMeta(SizeType size) noexcept {
        return kSmallCapacity - size;
    }

    static constexpr SizeType DecodeSmallSizeFromMeta(WordType meta) noexcept {
        return kSmallCapacity - static_cast<SizeType>(meta);
    }

    static constexpr SizeType TagMask() noexcept {
        if constexpr (kPayloadBits == 0) {
            return std::numeric_limits<SizeType>::max();
        } else if constexpr (config::kIsLittleEndian) {
            return ~CapacityMask();
        } else {
            return (SizeType{1} << kProbeBits) - 1;
        }
    }

    static constexpr SizeType CapacityMask() noexcept {
        if constexpr (kPayloadBits == 0) {
            return 0;
        } else if constexpr (config::kIsLittleEndian) {
            return (SizeType{1} << kPayloadBits) - 1;
        } else {
            return ~((SizeType{1} << kProbeBits) - 1);
        }
    }

    static SizeType PackCapacityWithTag(SizeType capacity, WordType tag) noexcept {
        assert(capacity <= max_external_capacity());
        if constexpr (config::kIsLittleEndian) {
            return (static_cast<SizeType>(tag) << kPayloadBits) | capacity;
        } else {
            return (capacity << kProbeBits) | static_cast<SizeType>(tag);
        }
    }

    static SizeType UnpackCapacity(SizeType packed) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return packed & CapacityMask();
        } else {
            return packed >> kProbeBits;
        }
    }

    static WordType UnpackTag(SizeType packed) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return static_cast<WordType>(packed >> kPayloadBits);
        } else {
            return static_cast<WordType>(packed & TagMask());
        }
    }

private:
    static void ClearStorage(StorageType& storage) noexcept {
        std::memset(storage.raw, 0, kStorageBytes);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_GENERIC_LAYOUT_POLICY_HPP
