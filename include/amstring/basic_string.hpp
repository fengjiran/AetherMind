// basic_string.hpp - Public API for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_BASIC_STRING_HPP
#define AETHERMIND_AMSTRING_BASIC_STRING_HPP

#include "core.hpp"
#include "string_fwd.hpp"
#include "stable_layout_policy.hpp"
#include "growth_policy.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace aethermind {

// Default layout policy selector
// Phase 1: All CharT use stable_layout_policy
// Phase 2: char switches to compact_layout_policy
template<typename CharT>
struct default_layout_policy {
    using type = StableLayoutPolicy<CharT>;
};

// basic_string - public API layer
// Standard-style interface with policy-based internal implementation
template<
    typename CharT,
    typename Traits,
    typename Allocator
>
class basic_string {
public:
    using value_type = CharT;
    using traits_type = Traits;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using reference = CharT&;
    using const_reference = const CharT&;
    using pointer = CharT*;
    using const_pointer = const CharT*;
    using iterator = CharT*;
    using const_iterator = const CharT*;

    static constexpr size_type npos = static_cast<size_type>(-1);

private:
    using layout_policy = typename default_layout_policy<CharT>::type;
    using growth_policy = default_growth_policy;
    using core_type = basic_string_core<CharT, Traits, Allocator, layout_policy, growth_policy>;

    core_type core_;

public:
    basic_string() noexcept = default;

    explicit basic_string(const allocator_type& a) noexcept
        : core_(a) {}

    basic_string(const CharT* s)
        : core_(s, traits_type::length(s)) {}

    basic_string(const CharT* s, size_type n)
        : core_(s, n) {}

    basic_string(std::basic_string_view<CharT, Traits> sv)
        : core_(sv.data(), sv.size()) {}

    basic_string(size_type count, CharT ch)
        : core_() {
        if (count > 0) {
            core_.reserve(count);
            traits_type::assign(core_.data(), count, ch);
            core_.data()[count] = CharT{};
            core_type::layout_policy_type::SetSize(core_.storage_, count);
        }
    }

    basic_string(const basic_string&) = default;
    basic_string(basic_string&&) noexcept = default;
    ~basic_string() = default;

    basic_string& operator=(const basic_string&) = default;
    basic_string& operator=(basic_string&&) noexcept = default;

    basic_string& operator=(const CharT* s) {
        core_.assign(s, traits_type::length(s));
        return *this;
    }

    basic_string& operator=(std::basic_string_view<CharT, Traits> sv) {
        core_.assign(sv.data(), sv.size());
        return *this;
    }

    basic_string& operator=(CharT ch) {
        core_.assign(&ch, 1);
        return *this;
    }

    const_pointer data() const noexcept { return core_.data(); }
    pointer data() noexcept { return core_.data(); }
    const_pointer c_str() const noexcept { return core_.c_str(); }

    size_type size() const noexcept { return core_.size(); }
    size_type length() const noexcept { return core_.size(); }
    size_type capacity() const noexcept { return core_.capacity(); }
    bool empty() const noexcept { return core_.empty(); }

    iterator begin() noexcept { return core_.data(); }
    iterator end() noexcept { return core_.data() + core_.size(); }
    const_iterator begin() const noexcept { return core_.data(); }
    const_iterator end() const noexcept { return core_.data() + core_.size(); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    reference operator[](size_type pos) noexcept {
        return core_.data()[pos];
    }

    const_reference operator[](size_type pos) const noexcept {
        return core_.data()[pos];
    }

    reference at(size_type pos) {
        if (pos >= core_.size()) {
            throw std::out_of_range("basic_string::at");
        }
        return core_.data()[pos];
    }

    const_reference at(size_type pos) const {
        if (pos >= core_.size()) {
            throw std::out_of_range("basic_string::at");
        }
        return core_.data()[pos];
    }

    reference front() noexcept { return *core_.data(); }
    const_reference front() const noexcept { return *core_.data(); }
    reference back() noexcept { return core_.data()[core_.size() - 1]; }
    const_reference back() const noexcept { return core_.data()[core_.size() - 1]; }

    void clear() noexcept { core_.clear(); }

    void reserve(size_type new_cap = 0) { core_.reserve(new_cap); }

    void resize(size_type count) { core_.resize(count); }
    void resize(size_type count, CharT ch) { core_.resize(count, ch); }

    void shrink_to_fit() { core_.shrink_to_fit(); }

    void push_back(CharT ch) { core_.append(&ch, 1); }

    void pop_back() {
        const size_type sz = core_.size();
        if (sz > 0) {
            core_.resize(sz - 1);
        }
    }

    basic_string& append(const basic_string& str) {
        core_.append(str.data(), str.size());
        return *this;
    }

    basic_string& append(const CharT* s) {
        core_.append(s, traits_type::length(s));
        return *this;
    }

    basic_string& append(const CharT* s, size_type n) {
        core_.append(s, n);
        return *this;
    }

    basic_string& append(size_type count, CharT ch) {
        if (count > 0) {
            core_.reserve(core_.size() + count);
            traits_type::assign(core_.data() + core_.size(), count, ch);
            core_.data()[core_.size() + count] = CharT{};
            core_type::layout_policy_type::SetSize(core_.storage_, core_.size() + count);
        }
        return *this;
    }

    basic_string& append(std::basic_string_view<CharT, Traits> sv) {
        core_.append(sv.data(), sv.size());
        return *this;
    }

    basic_string& operator+=(const basic_string& str) {
        return append(str);
    }

    basic_string& operator+=(const CharT* s) {
        return append(s);
    }

    basic_string& operator+=(CharT ch) {
        push_back(ch);
        return *this;
    }

    basic_string& operator+=(std::basic_string_view<CharT, Traits> sv) {
        return append(sv);
    }

    void swap(basic_string& other) noexcept {
        core_.swap(other.core_);
    }

    allocator_type get_allocator() const noexcept {
        return core_.get_allocator();
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_BASIC_STRING_HPP