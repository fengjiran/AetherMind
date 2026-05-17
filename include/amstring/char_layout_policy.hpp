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

    struct DecodedProbe {
        Category category;
        SizeType size;
        SizeType capacity;
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
    static constexpr SizeType kCapacityMask = config::kIsLittleEndian
                                                      ? ~(static_cast<SizeType>(kCategoryMask) << kTagBitShift)
                                                      : ~static_cast<SizeType>(kCategoryMask);

    AM_NODISCARD static bool is_small(const Storage& storage) noexcept {
        return DecodeProbe(storage).category == Category::kSmall;
    }

    AM_NODISCARD static bool is_small(const DecodedProbe& decoded) noexcept {
        return decoded.category == Category::kSmall;
    }

    AM_NODISCARD static bool is_external(const Storage& storage) noexcept {
        return DecodeProbe(storage).category == Category::kExternal;
    }

    AM_NODISCARD static bool is_external(const DecodedProbe& decoded) noexcept {
        return decoded.category == Category::kExternal;
    }

    AM_NODISCARD static Category category(const Storage& storage) noexcept {
        return DecodeProbe(storage).category;
    }

    AM_NODISCARD static DecodedProbe DecodeProbe(const Storage& storage) noexcept {
        const auto probe = GetProbe(storage);
        // clang-format off
        if constexpr (config::kIsLittleEndian) {
            // Small: probe ∈ [0, 23] ⇒ tag=kSmallTag AND size≤kSmallCapacity (single cmp folds both checks)
            if (probe < kStorageBytes) AM_LIKELY {
                // NOLINTNEXTLINE(modernize-use-designated-initializers)
                return {.category = Category::kSmall,
                        .size = kSmallCapacity - probe,
                        .capacity = kSmallCapacity};
            }

            // External: probe & 0xC0 == 0x80
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if ((probe & kCategoryMask) == kExternalTag) AM_UNLIKELY {
                // NOLINTNEXTLINE(modernize-use-designated-initializers)
                return {.category = Category::kExternal,
                        .size = storage.external.size,
                        .capacity = UnpackCapacity(storage.external.capacity_with_tag)};
            }
            // NOLINTNEXTLINE(modernize-use-designated-initializers)
            return {.category = Category::kInvalid, .size = 0, .capacity = 0};
        } else {
            const auto tag = TagFromProbe(probe);
            if (tag == kSmallTag) AM_LIKELY {
                const auto size = DecodeSmallSizeFromMeta(probe);
                if (size <= kSmallCapacity) AM_LIKELY {
                    // NOLINTNEXTLINE(modernize-use-designated-initializers)
                    return {.category = Category::kSmall, .size = size, .capacity = kSmallCapacity};
                }
                // NOLINTNEXTLINE(modernize-use-designated-initializers)
                return {.category = Category::kInvalid, .size = 0, .capacity = 0};
            }

            if (tag == kExternalTag) AM_UNLIKELY {
                // NOLINTNEXTLINE(modernize-use-designated-initializers)
                return {.category = Category::kExternal,
                        .size = storage.external.size,
                        .capacity = UnpackCapacity(storage.external.capacity_with_tag)};
            }

            // NOLINTNEXTLINE(modernize-use-designated-initializers)
            return {.category = Category::kInvalid, .size = 0, .capacity = 0};
        }
        // clang-format on
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
        return DecodeProbe(storage).category == Category::kSmall ? storage.small : storage.external.data;
    }

    AM_NODISCARD static const char* data(const Storage& storage, const DecodedProbe& decoded) noexcept {
        if (decoded.category == Category::kSmall) {
            return storage.small;
        }

        if (decoded.category == Category::kExternal) {
            return storage.external.data;
        }
        return nullptr;
    }

    AM_NODISCARD static char* data(Storage& storage) noexcept {
        return DecodeProbe(storage).category == Category::kSmall ? storage.small : storage.external.data;
    }

    AM_NODISCARD static char* data(Storage& storage, const DecodedProbe& decoded) noexcept {
        if (decoded.category == Category::kSmall) {
            return storage.small;
        }

        if (decoded.category == Category::kExternal) {
            return storage.external.data;
        }
        return nullptr;
    }

    AM_NODISCARD static SizeType size(const Storage& storage) noexcept {
        return DecodeProbe(storage).size;
    }

    AM_NODISCARD static SizeType size(const DecodedProbe& decoded) noexcept {
        return decoded.size;
    }

    AM_NODISCARD static SizeType capacity(const Storage& storage) noexcept {
        return DecodeProbe(storage).capacity;
    }

    AM_NODISCARD static SizeType capacity(const DecodedProbe& decoded) noexcept {
        return decoded.capacity;
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
        storage.small[size] = char{};
        SetSmallProbeUnchecked(storage, EncodeSmallSizeToProbe(size));
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
        AM_CHECK(size <= kSmallCapacity);
        storage.small[size] = char{};
        SetSmallProbeUnchecked(storage, EncodeSmallSizeToProbe(size));
    }

    static void SetExternalSize(Storage& storage, SizeType size) noexcept {
        AM_CHECK(size <= capacity(storage));
        storage.external.size = size;
        storage.external.data[size] = char{};
    }

    static void SetExternalCapacity(Storage& storage, SizeType capacity) noexcept {
        AM_CHECK(storage.external.size <= capacity);
        AM_CHECK(capacity <= max_external_capacity());
        storage.external.capacity_with_tag = PackCapacityWithTag(capacity);
    }

    static bool TryPushBackInplace(Storage& storage, char ch) noexcept {
        return TryPushBackInplace(storage, ch, DecodeProbe(storage));
    }

    static bool TryPushBackInplace(Storage& storage, char ch, const DecodedProbe& decoded) noexcept {
        // clang-format off
        if (decoded.category == Category::kExternal) AM_LIKELY {
            const SizeType size = decoded.size;
            if (size >= decoded.capacity) AM_UNLIKELY {
                return false;
            }

            storage.external.data[size] = ch;
            storage.external.data[size + 1] = char{};
            storage.external.size = size + 1;
            return true;
        }

        if (decoded.category != Category::kSmall) AM_UNLIKELY {
            return false;
        }

        const SizeType size = decoded.size;
        if (size >= kSmallCapacity) AM_UNLIKELY {
            return false;
        }
        // clang-format on

        storage.small[size] = ch;
        storage.small[size + 1] = char{};
        SetSmallProbeUnchecked(storage, EncodeSmallSizeToProbe(size + 1));
        return true;
    }

    static void CheckInvariants(const Storage& storage) noexcept {
        switch (const DecodedProbe decoded = DecodeProbe(storage); decoded.category) {
            case Category::kSmall:
                CheckSmallInvariants(storage, decoded);
                return;
            case Category::kExternal:
                CheckExternalInvariants(storage, decoded);
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

    static void SetSmallProbeUnchecked(Storage& storage, SizeType small_meta) noexcept {
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
        return kCapacityMask;
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
            return capacity_with_tag & kCapacityMask;
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

    static void CheckSmallInvariants(const Storage& storage, const DecodedProbe& decoded) noexcept {
        AM_DCHECK(decoded.category == Category::kSmall);
        AM_DCHECK(decoded.size <= kSmallCapacity);
        AM_DCHECK(decoded.capacity == kSmallCapacity);
        CheckDataInvariants(storage.small, decoded.size, decoded.capacity);
    }

    static void CheckExternalInvariants(const Storage& storage, const DecodedProbe& decoded) noexcept {
        AM_DCHECK(decoded.category == Category::kExternal);
        AM_DCHECK(UnpackTag(storage.external.capacity_with_tag) == kExternalTag);
        AM_DCHECK(storage.external.data != nullptr);
        AM_DCHECK(decoded.size <= decoded.capacity);
        CheckDataInvariants(storage.external.data, decoded.size, decoded.capacity);
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_CHAR_LAYOUT_POLICY_HPP
