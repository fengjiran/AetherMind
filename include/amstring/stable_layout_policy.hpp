// stable_layout_policy.hpp - Stable layout policy for multi-CharT support
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP

#include "invariant.hpp"

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace aethermind {

// Stable layout policy - generic, endian-independent layout
// Suitable for: char, char8_t, char16_t, char32_t, wchar_t
//
// Design principles:
// - Metadata stored in last CharT slot (not last byte)
// - Endian-independent encoding
// - Readable and testable across all CharT types
template<typename CharT>
struct StableLayoutPolicy {
    using value_type = CharT;
    using size_type = std::size_t;

    // Heap representation: 3 pointers/size fields
    struct MediumLarge {
        CharT* data;
        size_type size;
        size_type capacity_with_tag;
    };

    // Total storage budget equals MediumLarge size
    static constexpr std::size_t kStorageBytes = sizeof(MediumLarge);
    static constexpr std::size_t kSmallArraySize = kStorageBytes / sizeof(CharT);
    static constexpr std::size_t kSmallCapacity = kSmallArraySize - 1;

    // Storage type: union of small array and heap representation
    union StorageType {
        CharT small[kSmallArraySize];
        MediumLarge heap;
    };

    // Policy characteristics
    static constexpr bool kSupportsCompactBytePacking = false;
    static constexpr bool kEndianAware = false;

    // Initialize empty small string
    static void init_empty(StorageType& storage) noexcept {
        storage.small[0] = CharT{};
        storage.small[kSmallCapacity] = static_cast<CharT>(kSmallCapacity);
    }

    // Check if in small mode
    static bool is_small(const StorageType& storage) noexcept {
        return storage.small[kSmallCapacity] != CharT{};
    }

    // Check if in heap mode
    static bool is_heap(const StorageType& storage) noexcept {
        return !is_small(storage);
    }

    // Get data pointer
    static const CharT* data(const StorageType& storage) noexcept {
        return is_small(storage) ? storage.small : storage.heap.data;
    }

    static CharT* data(StorageType& storage) noexcept {
        return is_small(storage) ? storage.small : storage.heap.data;
    }

    // Get size
    static size_type size(const StorageType& storage) noexcept {
        if (is_small(storage)) {
            return kSmallCapacity - static_cast<size_type>(storage.small[kSmallCapacity]);
        }
        return storage.heap.size;
    }

    // Get capacity (actual character capacity, excluding tag bits)
    static size_type capacity(const StorageType& storage) noexcept {
        if (is_small(storage)) {
            return kSmallCapacity;
        }
        return storage.heap.capacity_with_tag;
    }

    // Set size
    static void set_size(StorageType& storage, size_type n) noexcept {
        if (is_small(storage)) {
            storage.small[kSmallCapacity] = static_cast<CharT>(kSmallCapacity - n);
        } else {
            storage.heap.size = n;
        }
    }

    // Set capacity (heap mode only)
    static void set_capacity(StorageType& storage, size_type n) noexcept {
        if (!is_small(storage)) {
            storage.heap.capacity_with_tag = n;
        }
    }

    // Initialize small from source
    static void init_small(
        StorageType& storage,
        const CharT* src,
        size_type n
    ) noexcept {
        std::memcpy(storage.small, src, n * sizeof(CharT));
        storage.small[n] = CharT{};
        storage.small[kSmallCapacity] = static_cast<CharT>(kSmallCapacity - n);
    }

    // Initialize heap from allocated pointer
    static void init_heap(
        StorageType& storage,
        CharT* ptr,
        size_type sz,
        size_type cap
    ) noexcept {
        storage.heap.data = ptr;
        storage.heap.size = sz;
        storage.heap.capacity_with_tag = cap;
    }

    // Get heap pointer
    static CharT* heap_ptr(StorageType& storage) noexcept {
        return storage.heap.data;
    }

    static const CharT* heap_ptr(const StorageType& storage) noexcept {
        return storage.heap.data;
    }

    // Destroy heap (caller must deallocate)
    static void destroy_heap(StorageType& storage) noexcept {
        storage.heap.data = nullptr;
        storage.heap.size = 0;
        storage.heap.capacity_with_tag = 0;
    }

    // Check invariants
    static void check_invariants(const StorageType& storage) noexcept {
        check_invariant_impl(data(storage), size(storage), capacity(storage));
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP