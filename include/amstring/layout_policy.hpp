// layout_policy.hpp - LayoutPolicy interface definition for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_LAYOUT_POLICY_HPP
#define AETHERMIND_AMSTRING_LAYOUT_POLICY_HPP

#include "generic_layout_policy.hpp"

#include <concepts>
#include <cstddef>

namespace aethermind {

// LayoutPolicy defines only storage layout primitives. It does not allocate,
// deallocate, choose growth capacity, or implement container algorithms.
template<typename Policy, typename CharT>
concept AmStringLayoutPolicy = requires(typename Policy::Storage& storage,
                                        const typename Policy::Storage& const_storage,
                                        const CharT* const_ptr,
                                        CharT* ptr,
                                        typename Policy::SizeType size) {
    typename Policy::ValueType;
    typename Policy::Storage;
    typename Policy::SizeType;
    typename Policy::Category;

    requires std::same_as<typename Policy::ValueType, CharT>;
    requires std::integral<typename Policy::SizeType>;
    requires std::convertible_to<decltype(Policy::kSmallCapacity), typename Policy::SizeType>;

    { Policy::is_small(const_storage) } noexcept -> std::same_as<bool>;
    { Policy::is_external(const_storage) } noexcept -> std::same_as<bool>;
    { Policy::category(const_storage) } noexcept -> std::same_as<typename Policy::Category>;

    { Policy::data(const_storage) } noexcept -> std::same_as<const CharT*>;
    { Policy::data(storage) } noexcept -> std::same_as<CharT*>;

    { Policy::size(const_storage) } noexcept -> std::same_as<typename Policy::SizeType>;
    { Policy::capacity(const_storage) } noexcept -> std::same_as<typename Policy::SizeType>;
    { Policy::max_external_capacity() } noexcept -> std::same_as<typename Policy::SizeType>;

    { Policy::InitEmpty(storage) } noexcept -> std::same_as<void>;
    { Policy::InitSmall(storage, const_ptr, size) } noexcept -> std::same_as<void>;
    { Policy::InitExternal(storage, ptr, size, size) } noexcept -> std::same_as<void>;

    { Policy::SetSmallSize(storage, size) } noexcept -> std::same_as<void>;
    { Policy::SetExternalSize(storage, size) } noexcept -> std::same_as<void>;
    { Policy::SetExternalCapacity(storage, size) } noexcept -> std::same_as<void>;

    { Policy::CheckInvariants(const_storage) } noexcept -> std::same_as<void>;
};

template<typename CharT>
struct DefaultLayoutPolicy {
    using type = GenericLayoutPolicy<CharT>;
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_LAYOUT_POLICY_HPP
