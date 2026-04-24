// stable_layout_policy.hpp - Stable layout policy for multi-CharT support
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP

#include "invariant.hpp"
#include "utils/logging.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace aethermind {

template<size_t N>
struct uint_of_size;

template<>
struct uint_of_size<1> {
    using type = uint8_t;
};

template<>
struct uint_of_size<2> {
    using type = uint16_t;
};

template<>
struct uint_of_size<4> {
    using type = uint32_t;
};

template<>
struct uint_of_size<8> {
    using type = uint64_t;
};

template<typename CharT>
using meta_word_t = uint_of_size<sizeof(CharT)>::type;

// Stable layout policy - generic, endian-independent layout
// Suitable for: char, char8_t, char16_t, char32_t, wchar_t
//
// Design principles:
// - Metadata stored in last CharT slot (not last byte)
// - Endian-independent encoding
// - Readable and testable across all CharT types
template<typename CharT>
struct StableLayoutPolicy {
    static_assert(std::is_trivial_v<CharT>, "CharT must be trivial.");
    static_assert(std::is_standard_layout_v<CharT>, "CharT must be standard layout.");
    static_assert(sizeof(size_t) >= sizeof(CharT));
    static_assert(std::is_same_v<CharT, char> || std::is_same_v<CharT, char8_t> || std::is_same_v<CharT, char16_t> ||
                          std::is_same_v<CharT, char32_t> || std::is_same_v<CharT, wchar_t>, "Unsupported CharT.");

    using value_type = CharT;
    using size_type = size_t;
    using word_type = meta_word_t<CharT>;

    struct MediumLarge {
        CharT* data;
        size_type size;
        size_type capacity_with_tag;
    };

    union Storage {
        CharT small[sizeof(MediumLarge) / sizeof(CharT)];
        MediumLarge ml;
        std::byte bytes[sizeof(MediumLarge)]{};
    };

    enum class Category : uint8_t {
        kSmall,
        kMedium,
        kLarge,
        kInvalid
    };

    // Total storage budget equals MediumLarge size
    static constexpr size_type kStorageBytes = sizeof(MediumLarge);
    static constexpr size_type kSmallSlots = kStorageBytes / sizeof(CharT);
    static constexpr size_type kSmallCapacity = kSmallSlots - 1;
    static constexpr size_type kWordBits = sizeof(size_type) * 8;// 64
    static constexpr size_type kProbeBits = sizeof(CharT) * 8;   // 8 / 16 / 32
    static constexpr size_type kPayloadBits = kWordBits - kProbeBits;
    static constexpr size_type kProbeByteOffset = kStorageBytes - sizeof(CharT);
    static constexpr word_type kMediumTag = static_cast<word_type>(kSmallCapacity + 1);
    static constexpr word_type kLargeTag = static_cast<word_type>(kSmallCapacity + 2);
    static constexpr size_type kCapacityMask = config::kIsLittleEndian ? (1ull << kPayloadBits) - 1 : ~((1ull << kProbeBits) - 1);
    static constexpr size_type kTagMask = config::kIsLittleEndian ? ~((1ull << kPayloadBits) - 1) : (1ull << kProbeBits) - 1;
    static constexpr bool kSupportsCompactBytePacking = false;
    static constexpr bool kEndianAware = true;

    static_assert(static_cast<size_type>(kMediumTag) > kSmallCapacity);
    static_assert(static_cast<size_type>(kLargeTag) > kSmallCapacity);

    static void ClearStorage(Storage& storage) noexcept {
        std::memset(storage.bytes, 0, kStorageBytes);
    }

    static word_type GetProbe(const Storage& storage) noexcept {
        word_type probe{};
        std::memcpy(&probe, storage.bytes + kProbeByteOffset, sizeof(word_type));
        return probe;
    }

    static void SetProbe(Storage& storage, word_type probe) noexcept {
        std::memcpy(storage.bytes + kProbeByteOffset, &probe, sizeof(word_type));
    }

    static constexpr size_type MaxMLCapacity() noexcept {
        if constexpr (kPayloadBits == 0) {
            return 0;
        } else if constexpr (kPayloadBits >= kWordBits) {
            return std::numeric_limits<size_type>::max();
        } else {
            return (1ULL << kPayloadBits) - 1;
        }
    }

    static bool IsSmall(const Storage& storage) noexcept {
        return GetProbe(storage) <= static_cast<word_type>(kSmallCapacity);
    }

    static bool IsMedium(const Storage& storage) noexcept {
        return GetProbe(storage) == kMediumTag;
    }

    static bool IsLarge(const Storage& storage) noexcept {
        return GetProbe(storage) == kLargeTag;
    }

    static Category GetCategory(const Storage& storage) noexcept {
        const word_type probe = GetProbe(storage);
        if (probe <= static_cast<word_type>(kSmallCapacity)) {
            return Category::kSmall;
        }

        if (probe == kMediumTag) {
            return Category::kMedium;
        }

        if (probe == kLargeTag) {
            return Category::kLarge;
        }

        return Category::kInvalid;
    }

    static size_type PackCapacityWithTag(size_type capacity, CharT tag) noexcept {
        AM_CHECK(capacity <= MaxMLCapacity());
        if constexpr (config::kIsLittleEndian) {
            return (static_cast<size_type>(tag) << kPayloadBits) | capacity;
        } else {
            return (capacity << kProbeBits) | static_cast<size_type>(tag);
        }
    }

    static word_type UnpackTag(size_type capacity_with_tag) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return static_cast<word_type>(capacity_with_tag >> kPayloadBits);
        } else {
            return static_cast<word_type>(capacity_with_tag & kTagMask);
        }
    }

    static size_type UnpackCapacity(size_type capacity_with_tag) noexcept {
        if constexpr (config::kIsLittleEndian) {
            return capacity_with_tag & kCapacityMask;
        } else {
            return capacity_with_tag >> kProbeBits;
        }
    }

    // Initialize empty small string
    static void InitEmpty(Storage& storage) noexcept {
        ClearStorage(storage);
        SetSize(storage, 0);
    }

    // Get data pointer
    static const CharT* data(const Storage& storage) noexcept {
        return IsSmall(storage) ? storage.small : storage.ml.data;
    }

    static CharT* data(Storage& storage) noexcept {
        return IsSmall(storage) ? storage.small : storage.ml.data;
    }

    // Get size
    static size_type size(const Storage& storage) noexcept {
        if (IsSmall(storage)) {
            return kSmallCapacity - static_cast<size_type>(storage.small[kSmallCapacity]);
        }
        return storage.ml.size;
    }

    // Get capacity (actual character capacity, excluding tag bits)
    static size_type capacity(const Storage& storage) noexcept {
        if (IsSmall(storage)) {
            return kSmallCapacity;
        }
        return UnpackCapacity(storage.ml.capacity_with_tag);
    }

    // Set size
    static void SetSize(Storage& storage, size_type sz) noexcept {
        AM_CHECK(sz <= capacity(storage));
        if (IsSmall(storage)) {
            storage.small[sz] = CharT{};
            const auto probe = static_cast<word_type>(kSmallCapacity - sz);
            std::memcpy(storage.bytes + kProbeByteOffset, &probe, sizeof(word_type));
        } else {
            storage.ml.data[sz] = CharT{};
            storage.ml.size = sz;
        }
    }

    static void SetCapacity(Storage& storage, size_type cap) noexcept {
        if (!IsSmall(storage)) {
            const auto tag = static_cast<CharT>(UnpackTag(storage.ml.capacity_with_tag));
            storage.ml.capacity_with_tag = PackCapacityWithTag(cap, tag);
        }
    }

    // Initialize small from source
    static void InitSmall(Storage& storage, const CharT* src, size_type n) noexcept {
        AM_CHECK(n <= kSmallCapacity);
        ClearStorage(storage);
        if (src != nullptr && n > 0) {
            std::memcpy(static_cast<void*>(storage.small), static_cast<const void*>(src), n * sizeof(CharT));
        }

        SetSize(storage, n);
    }

    // Initialize heap from allocated pointer
    static void init_heap(Storage& storage, CharT* ptr, size_type sz, size_type cap) noexcept {
        AM_CHECK(ptr != nullptr);
        AM_CHECK(sz <= cap);
        storage.ml.data = ptr;
        storage.ml.size = sz;
        storage.ml.capacity_with_tag = cap;
    }

    // Get heap pointer
    static CharT* heap_ptr(Storage& storage) noexcept {
        return storage.ml.data;
    }

    static const CharT* heap_ptr(const Storage& storage) noexcept {
        return storage.ml.data;
    }

    // Destroy heap (caller must deallocate)
    static void destroy_heap(Storage& storage) noexcept {
        storage.ml.data = nullptr;
        storage.ml.size = 0;
        storage.ml.capacity_with_tag = 0;
    }

    // Check invariants
    static void check_invariants(const Storage& storage) noexcept {
        check_invariant_impl(data(storage), size(storage), capacity(storage));
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP
