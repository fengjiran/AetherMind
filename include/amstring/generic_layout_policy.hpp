// generic_layout_policy.hpp - Generic layout policy for multi-CharT amstring support
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_GENERIC_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_GENERIC_LAYOUT_POLICY_HPP

#include "config.hpp"
#include "invariant.hpp"
#include "macros.h"
#include "utils/logging.h"

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
    static_assert(sizeof(std::size_t) >= sizeof(CharT), "CharT cannot be wider than size_t.");
    static_assert(std::is_same_v<CharT, char> || std::is_same_v<CharT, char8_t> || std::is_same_v<CharT, char16_t> ||
                          std::is_same_v<CharT, char32_t> || std::is_same_v<CharT, wchar_t>,
                  "Unsupported CharT.");

    using ValueType = CharT;
    using SizeType = std::size_t;
    using ProbeWordType = MetaWordT<CharT>;

    // capacity_with_tag layout (Little-Endian, e.g. CharT=char, kProbeBits=8):
    //   [63 ... kPayloadBits | kPayloadBits-1 ... 0]
    //   [      tag (8b)      |     capacity (56b)    ]
    //
    // capacity_with_tag layout (Big-Endian):
    //   [63 ... kProbeBits | kProbeBits-1 ... 0]
    //   [  capacity (56b)  |     tag (8b)       ]
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

    // Storage category determined by probe value:
    //   kSmall    - probe ∈ [0, kSmallCapacity]; data stored inline in small[]
    //   kExternal - probe == kExternalTag; data stored on heap
    //   kInvalid  - probe value doesn't match any valid encoding (data corruption)
    enum class Category : std::uint8_t {
        kSmall,
        kExternal,
        kInvalid
    };

    static_assert(sizeof(Storage) == sizeof(ExternalRep));

    static constexpr SizeType kStorageBytes = sizeof(ExternalRep);
    static constexpr SizeType kSmallSlots = kStorageBytes / sizeof(CharT);
    static constexpr SizeType kSmallCapacity = kSmallSlots - 1;
    static constexpr SizeType kProbeBits = sizeof(CharT) * 8;
    static constexpr SizeType kSizeTypeBits = sizeof(SizeType) * 8;// 64
    static constexpr SizeType kPayloadBits = kSizeTypeBits - kProbeBits;
    static constexpr SizeType kProbeByteOffset = kStorageBytes - sizeof(CharT);
    static constexpr ProbeWordType kExternalTag = static_cast<ProbeWordType>(kSmallCapacity + 1);

    AM_NODISCARD static bool is_small(const Storage& storage) noexcept {
        return GetProbe(storage) <= static_cast<ProbeWordType>(kSmallCapacity);
    }

    AM_NODISCARD static bool is_external(const Storage& storage) noexcept {
        return GetProbe(storage) == kExternalTag;
    }

    AM_NODISCARD static Category category(const Storage& storage) noexcept {
        const auto meta = GetProbe(storage);
        if (meta <= static_cast<ProbeWordType>(kSmallCapacity)) {
            return Category::kSmall;
        }

        if (meta == kExternalTag) {
            return Category::kExternal;
        }

        return Category::kInvalid;
    }

    AM_NODISCARD static constexpr SizeType max_external_capacity() noexcept {
        if constexpr (kPayloadBits == 0) {
            return 0;
        } else if constexpr (kPayloadBits >= kSizeTypeBits) {
            return std::numeric_limits<SizeType>::max();
        } else {
            return (SizeType{1} << kPayloadBits) - 1;
        }
    }

    AM_NODISCARD static const CharT* data(const Storage& storage) noexcept {
        return is_small(storage) ? storage.small : storage.external.data;
    }

    AM_NODISCARD static CharT* data(Storage& storage) noexcept {
        return is_small(storage) ? storage.small : storage.external.data;
    }

    AM_NODISCARD static SizeType size(const Storage& storage) noexcept {
        const auto probe = GetProbe(storage);
        if (probe <= static_cast<ProbeWordType>(kSmallCapacity)) {
            return DecodeSmallSizeFromProbe(probe);
        }
        return storage.external.size;
    }

    AM_NODISCARD static SizeType capacity(const Storage& storage) noexcept {
        if (is_small(storage)) {
            return kSmallCapacity;
        }
        return UnpackCapacity(storage.external.capacity_with_tag);
    }

    static void InitEmpty(Storage& storage) noexcept {
        ClearStorage(storage);
        SetSmallSize(storage, 0);
    }

    static void InitSmall(Storage& storage, const CharT* src, SizeType size) noexcept {
        AM_CHECK(size <= kSmallCapacity);
        ClearStorage(storage);
        if (src != nullptr && size != 0) {
            std::memcpy(storage.small, src, size * sizeof(CharT));
        }
        SetSmallSize(storage, size);
    }

    static void InitExternal(Storage& storage, CharT* ptr, SizeType size, SizeType capacity) noexcept {
        AM_CHECK(ptr != nullptr);
        AM_CHECK(size <= capacity);
        AM_CHECK(capacity <= max_external_capacity());
        storage.external.data = ptr;
        storage.external.size = size;
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity, kExternalTag);
        ptr[size] = CharT{};
    }

    // Design constraint: when size == kSmallCapacity, the null terminator
    // at small[size] occupies the same memory as the probe byte.
    // EncodeSmallSizeToProbe(kSmallCapacity) == 0 == CharT{}, so both
    // writes produce the same value. This invariant must be preserved
    // if the probe encoding scheme is ever changed.
    static void SetSmallSize(Storage& storage, SizeType size) noexcept {
        AM_CHECK(is_small(storage));
        AM_CHECK(size <= kSmallCapacity);
        storage.small[size] = CharT{};
        SetProbe(storage, EncodeSmallSizeToProbe(size));
    }

    static void SetExternalSize(Storage& storage, SizeType size) noexcept {
        AM_CHECK(is_external(storage));
        AM_CHECK(size <= capacity(storage));
        storage.external.size = size;
        storage.external.data[size] = CharT{};
    }

    static void SetExternalCapacity(Storage& storage, SizeType capacity) noexcept {
        AM_CHECK(is_external(storage));
        AM_CHECK(storage.external.size <= capacity);
        AM_CHECK(capacity <= max_external_capacity());
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity, kExternalTag);
    }

    static void CheckInvariants(const Storage& storage) noexcept {
        switch (category(storage)) {
            case Category::kSmall:
                CheckSmallInvariants(storage);
                return;
            case Category::kExternal:
                CheckExternalInvariants(storage);
                return;
            case Category::kInvalid:
                AM_DCHECK(false);
                return;
        }
        AM_DCHECK(false);
        AM_UNREACHABLE();
    }

private:
    static void ClearStorage(Storage& storage) noexcept {
        std::memset(storage.raw, 0, kStorageBytes);
    }

    AM_NODISCARD static ProbeWordType GetProbe(const Storage& storage) noexcept {
        ProbeWordType probe{};
        std::memcpy(&probe, storage.raw + kProbeByteOffset, sizeof(ProbeWordType));
        return probe;
    }

    static void SetProbe(Storage& storage, ProbeWordType probe) noexcept {
        std::memcpy(storage.raw + kProbeByteOffset, &probe, sizeof(ProbeWordType));
    }

    AM_NODISCARD static SizeType PackCapacityWithTag(SizeType capacity, ProbeWordType tag) noexcept {
        AM_CHECK(capacity <= max_external_capacity());
        if constexpr (config::kIsLittleEndian) {
            return (static_cast<SizeType>(tag) << kPayloadBits) | capacity;
        } else {
            return (capacity << kProbeBits) | static_cast<SizeType>(tag);
        }
    }

    static void CheckSmallInvariants(const Storage& storage) noexcept {
        const auto probe = GetProbe(storage);
        const auto decoded_size = DecodeSmallSizeFromProbe(probe);

        AM_DCHECK(probe <= static_cast<ProbeWordType>(kSmallCapacity));
        AM_DCHECK(decoded_size <= kSmallCapacity);
        CheckDataInvariants(storage.small, decoded_size, kSmallCapacity);
    }

    static void CheckExternalInvariants(const Storage& storage) noexcept {
        const auto decoded_capacity = UnpackCapacity(storage.external.capacity_with_tag);

        AM_DCHECK(GetProbe(storage) == kExternalTag);
        AM_DCHECK(UnpackTag(storage.external.capacity_with_tag) == kExternalTag);
        AM_DCHECK(storage.external.size <= decoded_capacity);
        CheckDataInvariants(storage.external.data, storage.external.size, decoded_capacity);
    }

    AM_NODISCARD static constexpr SizeType TagMask() noexcept {
        if constexpr (kPayloadBits == 0) {
            return std::numeric_limits<SizeType>::max();
        } else if constexpr (config::kIsLittleEndian) {
            return ~CapacityMask();
        } else {
            return (SizeType{1} << kProbeBits) - 1;
        }
    }

    AM_NODISCARD static constexpr SizeType CapacityMask() noexcept {
        if constexpr (kPayloadBits == 0) {
            return 0;
        } else if constexpr (config::kIsLittleEndian) {
            return (SizeType{1} << kPayloadBits) - 1;
        } else {
            return ~((SizeType{1} << kProbeBits) - 1);
        }
    }

    // External-only helpers
    AM_NODISCARD static SizeType UnpackCapacity(SizeType capacity_with_tag) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return capacity_with_tag & CapacityMask();
        } else {
            return capacity_with_tag >> kProbeBits;
        }
    }

    // External-only helpers
    AM_NODISCARD static ProbeWordType UnpackTag(SizeType capacity_with_tag) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return static_cast<ProbeWordType>(capacity_with_tag >> kPayloadBits);
        } else {
            return static_cast<ProbeWordType>(capacity_with_tag & TagMask());
        }
    }

    // Inverted encoding: probe = kSmallCapacity - size
    // This makes is_small() a single comparison (probe <= kSmallCapacity),
    // and ensures InitEmpty produces probe == kSmallCapacity (within small range).
    AM_NODISCARD static constexpr SizeType EncodeSmallSizeToProbe(SizeType size) noexcept {
        return kSmallCapacity - size;
    }

    AM_NODISCARD static constexpr SizeType DecodeSmallSizeFromProbe(ProbeWordType probe) noexcept {
        return kSmallCapacity - static_cast<SizeType>(probe);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_GENERIC_LAYOUT_POLICY_HPP
