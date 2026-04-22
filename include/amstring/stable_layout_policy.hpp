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
struct stable_layout_policy {
    using value_type = CharT;
    using size_type = std::size_t;

    // Heap representation: 3 pointers/size fields
    struct heap_rep {
        CharT* data;
        size_type size;
        size_type capacity_with_tag;
    };

    // Total storage budget equals heap_rep size
    static constexpr std::size_t kStorageBytes = sizeof(heap_rep);
    static constexpr std::size_t kSmallArraySize = kStorageBytes / sizeof(CharT);
    static constexpr std::size_t kSmallCapacity = kSmallArraySize - 1;

    // Storage type: union of small array and heap representation
    union storage_type {
        CharT small[kSmallArraySize];
        heap_rep heap;
    };

    // Policy characteristics
    static constexpr bool kSupportsCompactBytePacking = false;
    static constexpr bool kEndianAware = false;

    // Initialize empty small string
    static void init_empty(storage_type& storage) noexcept {
        storage.small[0] = CharT{};
        storage.small[kSmallCapacity] = static_cast<CharT>(kSmallCapacity);
    }

    // Check if in small mode
    static bool is_small(const storage_type& storage) noexcept {
        return storage.small[kSmallCapacity] != CharT{};
    }

    // Check if in heap mode
    static bool is_heap(const storage_type& storage) noexcept {
        return !is_small(storage);
    }

    // Get data pointer
    static const CharT* data(const storage_type& storage) noexcept {
        return is_small(storage) ? storage.small : storage.heap.data;
    }

    static CharT* data(storage_type& storage) noexcept {
        return is_small(storage) ? storage.small : storage.heap.data;
    }

    // Get size
    static size_type size(const storage_type& storage) noexcept {
        if (is_small(storage)) {
            return kSmallCapacity - static_cast<size_type>(storage.small[kSmallCapacity]);
        }
        return storage.heap.size;
    }

    // Get capacity (actual character capacity, excluding tag bits)
    static size_type capacity(const storage_type& storage) noexcept {
        if (is_small(storage)) {
            return kSmallCapacity;
        }
        return storage.heap.capacity_with_tag;
    }

    // Set size
    static void set_size(storage_type& storage, size_type n) noexcept {
        if (is_small(storage)) {
            storage.small[kSmallCapacity] = static_cast<CharT>(kSmallCapacity - n);
        } else {
            storage.heap.size = n;
        }
    }

    // Set capacity (heap mode only)
    static void set_capacity(storage_type& storage, size_type n) noexcept {
        if (!is_small(storage)) {
            storage.heap.capacity_with_tag = n;
        }
    }

    // Initialize small from source
    static void init_small(
        storage_type& storage,
        const CharT* src,
        size_type n
    ) noexcept {
        std::memcpy(storage.small, src, n * sizeof(CharT));
        storage.small[n] = CharT{};
        storage.small[kSmallCapacity] = static_cast<CharT>(kSmallCapacity - n);
    }

    // Initialize heap from allocated pointer
    static void init_heap(
        storage_type& storage,
        CharT* ptr,
        size_type sz,
        size_type cap
    ) noexcept {
        storage.heap.data = ptr;
        storage.heap.size = sz;
        storage.heap.capacity_with_tag = cap;
    }

    // Get heap pointer
    static CharT* heap_ptr(storage_type& storage) noexcept {
        return storage.heap.data;
    }

    static const CharT* heap_ptr(const storage_type& storage) noexcept {
        return storage.heap.data;
    }

    // Destroy heap (caller must deallocate)
    static void destroy_heap(storage_type& storage) noexcept {
        storage.heap.data = nullptr;
        storage.heap.size = 0;
        storage.heap.capacity_with_tag = 0;
    }

    // Check invariants
    static void check_invariants(const storage_type& storage) noexcept {
        check_invariant_impl(data(storage), size(storage), capacity(storage));
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_STABLE_LAYOUT_POLICY_HPP