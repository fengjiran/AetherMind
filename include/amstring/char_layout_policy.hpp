// CharLayoutPolicy - char-specialized amstring storage layout.
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_CHAR_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_CHAR_LAYOUT_POLICY_HPP

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

/// Policy for the char-specialized 24-byte Small/External storage layout.
///
/// This policy is interface-compatible with GenericLayoutPolicy<char>, but it
/// uses a 2-bit category marker in the capacity word so External capacity keeps
/// a 62-bit payload on 64-bit platforms.
///
/// Responsibilities:
/// - Define char-only storage layout (union of inline array and external rep)
/// - Encode/decode Small size via the last byte of inline storage
/// - Pack/unpack External capacity with endian-aware 2-bit category marker
/// - Provide layout primitives: init, category detection, data/size/capacity access
///
/// Does NOT handle allocation, deallocation, growth policy, traits algorithms,
/// exception safety, or ownership semantics.
struct CharLayoutPolicy {
    using ValueType = char;
    using SizeType = std::size_t;

    struct ExternalRep {
        char* data;                // Borrowed; caller owns allocation and deallocation.
        SizeType size;             // Current length, excluding null terminator.
        SizeType capacity_with_tag;// Packed capacity plus 2-bit category marker.
    };

    union Storage {
        char small[sizeof(ExternalRep)];     // Inline buffer for Small state.
        ExternalRep external;                // Heap representation for External state.
        std::byte raw[sizeof(ExternalRep)]{};// Raw bytes for probe access.
    };

    enum class Category : std::uint8_t {
        kSmall,
        kExternal,
        kInvalid,
    };

    static_assert(sizeof(Storage) == sizeof(ExternalRep));
    static_assert(std::is_trivially_copyable_v<Storage>);

    static constexpr SizeType kStorageBytes = sizeof(ExternalRep);
    static constexpr SizeType kSmallSlots = kStorageBytes;
    static constexpr SizeType kSmallCapacity = kSmallSlots - 1;
    static constexpr SizeType kCategoryBits = 2;
    static constexpr SizeType kSizeTypeBits = sizeof(SizeType) * 8;// 64
    static constexpr SizeType kPayloadBits = kSizeTypeBits - kCategoryBits;
    static constexpr SizeType kProbeByteOffset = kStorageBytes - 1;
    static constexpr SizeType kTagBitShift = (sizeof(SizeType) - 1) * 8;

    static constexpr std::uint8_t kCategoryMask = config::kIsLittleEndian ? 0xC0U : 0x03U;
    static constexpr std::uint8_t kSmallMetaMask = config::kIsLittleEndian ? 0x3FU : 0xFCU;
    static constexpr std::uint8_t kSmallTag = 0x00U;
    static constexpr std::uint8_t kExternalTag = config::kIsLittleEndian ? 0x80U : 0x02U;

    AM_NODISCARD static bool is_small(const Storage& storage) noexcept {
        const auto probe = GetProbe(storage);
        return TagFromProbe(probe) == kSmallTag && DecodeSmallSizeFromMeta(probe) <= kSmallCapacity;
    }

    AM_NODISCARD static bool is_external(const Storage& storage) noexcept {
        return TagFromProbe(GetProbe(storage)) == kExternalTag;
    }

    AM_NODISCARD static Category category(const Storage& storage) noexcept {
        const auto probe = GetProbe(storage);
        const auto tag = TagFromProbe(probe);
        if (tag == kSmallTag && DecodeSmallSizeFromMeta(probe) <= kSmallCapacity) {
            return Category::kSmall;
        }

        if (tag == kExternalTag) {
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

    AM_NODISCARD static const char* data(const Storage& storage) noexcept {
        return is_small(storage) ? storage.small : storage.external.data;
    }

    AM_NODISCARD static char* data(Storage& storage) noexcept {
        return is_small(storage) ? storage.small : storage.external.data;
    }

    AM_NODISCARD static SizeType size(const Storage& storage) noexcept {
        if (const auto probe = GetProbe(storage); TagFromProbe(probe) == kSmallTag) {
            if (const auto decoded = DecodeSmallSizeFromMeta(probe); decoded <= kSmallCapacity) {
                return decoded;
            }
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

    static void InitSmall(Storage& storage, const char* src, SizeType size) noexcept {
        AM_CHECK(size <= kSmallCapacity);
        AM_CHECK(size == 0 || src != nullptr);
        ClearStorage(storage);
        if (src != nullptr && size != 0) {
            std::memcpy(storage.small, src, size);
        }
        SetSmallSize(storage, size);
    }

    static void InitExternal(Storage& storage, char* ptr, SizeType size, SizeType capacity) noexcept {
        AM_CHECK(ptr != nullptr);
        AM_CHECK(size <= capacity);
        AM_CHECK(capacity <= max_external_capacity());
        storage.external.data = ptr;
        storage.external.size = size;
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity);
        ptr[size] = char{};
    }

    static void SetSmallSize(Storage& storage, SizeType size) noexcept {
        AM_CHECK(is_small(storage));
        AM_CHECK(size <= kSmallCapacity);
        storage.small[size] = char{};
        SetSmallProbe(storage, EncodeSmallSizeToProbe(size));
    }

    static void SetExternalSize(Storage& storage, SizeType size) noexcept {
        AM_CHECK(is_external(storage));
        AM_CHECK(size <= capacity(storage));
        storage.external.size = size;
        storage.external.data[size] = char{};
    }

    static void SetExternalCapacity(Storage& storage, SizeType capacity) noexcept {
        AM_CHECK(is_external(storage));
        AM_CHECK(storage.external.size <= capacity);
        AM_CHECK(capacity <= max_external_capacity());
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity);
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

    AM_NODISCARD static std::uint8_t GetProbe(const Storage& storage) noexcept {
        return std::to_integer<std::uint8_t>(storage.raw[kProbeByteOffset]);
    }

    static void SetSmallProbe(Storage& storage, SizeType small_meta) noexcept {
        AM_CHECK(is_small(storage));
        AM_CHECK(small_meta <= kSmallCapacity);
        const auto encoded = config::kIsLittleEndian ? static_cast<std::uint8_t>(small_meta)
                                                     : static_cast<std::uint8_t>(small_meta << kCategoryBits);
        storage.raw[kProbeByteOffset] = static_cast<std::byte>(encoded);
    }

    AM_NODISCARD static constexpr SizeType EncodeSmallSizeToProbe(SizeType size) noexcept {
        return kSmallCapacity - size;
    }

    AM_NODISCARD static constexpr SizeType DecodeSmallSizeFromMeta(std::uint8_t probe) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return kSmallCapacity - static_cast<SizeType>(probe & kSmallMetaMask);
        } else {
            return kSmallCapacity - static_cast<SizeType>((probe & kSmallMetaMask) >> kCategoryBits);
        }
    }

    AM_NODISCARD static constexpr SizeType CapacityMask() noexcept {
        if constexpr (config::kIsLittleEndian) {
            return ~(static_cast<SizeType>(kCategoryMask) << kTagBitShift);
        } else {
            return ~static_cast<SizeType>(kCategoryMask);
        }
    }

    AM_NODISCARD static SizeType PackCapacityWithTag(SizeType capacity) noexcept {
        AM_CHECK(capacity <= max_external_capacity());
        if constexpr (config::kIsLittleEndian) {
            return capacity | (static_cast<SizeType>(kExternalTag) << kTagBitShift);
        } else {
            return (capacity << kCategoryBits) | static_cast<SizeType>(kExternalTag);
        }
    }

    AM_NODISCARD static SizeType UnpackCapacity(SizeType capacity_with_tag) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return capacity_with_tag & CapacityMask();
        } else {
            return capacity_with_tag >> kCategoryBits;
        }
    }

    AM_NODISCARD static std::uint8_t UnpackTag(SizeType capacity_with_tag) noexcept {
        if constexpr (config::kIsLittleEndian) {
            const auto probe = static_cast<std::uint8_t>(capacity_with_tag >> kTagBitShift);
            return TagFromProbe(probe);
        } else {
            return static_cast<std::uint8_t>(capacity_with_tag) & kCategoryMask;
        }
    }

private:
    static void ClearStorage(Storage& storage) noexcept {
        std::memset(storage.raw, 0, kStorageBytes);
    }

    AM_NODISCARD static constexpr std::uint8_t TagFromProbe(std::uint8_t probe) noexcept {
        return probe & kCategoryMask;
    }

    static void CheckSmallInvariants(const Storage& storage) noexcept {
        const auto probe = GetProbe(storage);
        const auto decoded_size = DecodeSmallSizeFromMeta(probe);

        AM_DCHECK(TagFromProbe(probe) == kSmallTag);
        AM_DCHECK(decoded_size <= kSmallCapacity);
        CheckDataInvariants(storage.small, decoded_size, kSmallCapacity);
    }

    static void CheckExternalInvariants(const Storage& storage) noexcept {
        const auto decoded_capacity = UnpackCapacity(storage.external.capacity_with_tag);

        AM_DCHECK(UnpackTag(storage.external.capacity_with_tag) == kExternalTag);
        AM_DCHECK(storage.external.data != nullptr);
        AM_DCHECK(storage.external.size <= decoded_capacity);
        CheckDataInvariants(storage.external.data, storage.external.size, decoded_capacity);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CHAR_LAYOUT_POLICY_HPP
