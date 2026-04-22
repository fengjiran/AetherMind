// compact_layout_policy.hpp - Compact layout policy for char optimization
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_COMPACT_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_COMPACT_LAYOUT_POLICY_HPP

#include "config.hpp"
#include "invariant.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace aethermind {

// Compact layout policy - fbstring-like 24-byte layout for char
// Only enabled for sizeof(CharT) == 1 (char, char8_t)
//
// Design goals:
// - 24-byte object size
// - 23-char SSO capacity
// - Byte-level category marker
// - Endian-aware encoding
//
// TODO: Implement in Phase 2 (Milestone 12)
// Phase 1 uses stable_layout_policy for all CharT

template<typename CharT>
struct compact_layout_policy {
    static_assert(sizeof(CharT) == 1, "compact_layout_policy only for 1-byte CharT");

    using value_type = CharT;
    using size_type = std::size_t;

    // Category enum
    enum class category : std::uint8_t {
        small = 0,
        heap  = 1,
        // large = 2 (reserved for future COW)
    };

    // Heap representation
    struct heap_rep {
        CharT* data;
        size_type size;
        size_type cap_with_category;
    };

    // Storage: 24 bytes total
    static constexpr std::size_t kStorageBytes = sizeof(heap_rep);
    static constexpr std::size_t kSmallCapacity = kStorageBytes - 1;

    union storage_type {
        CharT small[kStorageBytes];
        heap_rep heap;
    };

    static constexpr bool kSupportsCompactBytePacking = true;
    static constexpr bool kEndianAware = true;

    // Endian-aware constants
    static constexpr std::size_t kCategoryShift =
        config::kIsLittleEndian ? 0 : (sizeof(size_type) - 1) * 8;

    // Placeholder implementations for Phase 2
    static void init_empty(storage_type& storage) noexcept;
    static bool is_small(const storage_type& storage) noexcept;
    static bool is_heap(const storage_type& storage) noexcept;
    static const CharT* data(const storage_type& storage) noexcept;
    static CharT* data(storage_type& storage) noexcept;
    static size_type size(const storage_type& storage) noexcept;
    static size_type capacity(const storage_type& storage) noexcept;
    static void set_size(storage_type& storage, size_type n) noexcept;
    static void set_capacity(storage_type& storage, size_type n) noexcept;
    static void init_small(storage_type& storage, const CharT* src, size_type n) noexcept;
    static void init_heap(storage_type& storage, CharT* ptr, size_type sz, size_type cap) noexcept;
    static void destroy_heap(storage_type& storage) noexcept;
    static void check_invariants(const storage_type& storage) noexcept;
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_COMPACT_LAYOUT_POLICY_HPP