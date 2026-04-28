// basic_string.hpp - Public API for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_BASIC_STRING_HPP
#define AETHERMIND_AMSTRING_BASIC_STRING_HPP

#include "core.hpp"
#include "growth_policy.hpp"
#include "string_fwd.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace aethermind {

// BasicString - public API layer
// Standard-style interface with policy-based internal implementation
template<typename CharT, typename Traits, typename Allocator>
class BasicString {
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
    using LayoutPolicy = DefaultLayoutPolicy<CharT>::type;
    using GrowthPolicy = DefaultGrowthPolicy;
    using CoreType = BasicStringCore<CharT, Traits, Allocator, LayoutPolicy, GrowthPolicy>;

    CoreType core_;

public:
    BasicString() noexcept = default;
    BasicString(const BasicString&) = default;
    BasicString(BasicString&&) noexcept = default;
    ~BasicString() = default;

    explicit BasicString(const allocator_type& a) noexcept : core_(a) {}

    BasicString(const CharT* s) : core_(s, traits_type::length(s)) {}

    BasicString(const CharT* s, size_type n) : core_(s, n) {}

    BasicString(std::basic_string_view<CharT, Traits> sv) : core_(sv.data(), sv.size()) {}

    BasicString(size_type count, CharT ch) {
        core_.resize(count, ch);
    }

    BasicString& operator=(const BasicString&) = default;
    BasicString& operator=(BasicString&&) noexcept = default;
    BasicString& operator=(const CharT* s) {
        core_.assign(s, traits_type::length(s));
        return *this;
    }

    BasicString& operator=(std::basic_string_view<CharT, Traits> sv) {
        core_.assign(sv);
        return *this;
    }

    BasicString& operator=(CharT ch) {
        core_.assign(1, ch);
        return *this;
    }

    const_pointer data() const noexcept {
        return core_.data();
    }

    pointer data() noexcept {
        return core_.data();
    }

    const_pointer c_str() const noexcept {
        return core_.c_str();
    }

    size_type size() const noexcept {
        return core_.size();
    }

    size_type length() const noexcept {
        return core_.size();
    }

    size_type capacity() const noexcept {
        return core_.capacity();
    }

    bool empty() const noexcept {
        return core_.empty();
    }

    iterator begin() noexcept {
        return core_.data();
    }

    iterator end() noexcept {
        return core_.data() + core_.size();
    }

    const_iterator begin() const noexcept {
        return core_.data();
    }

    const_iterator end() const noexcept {
        return core_.data() + core_.size();
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }

    const_iterator cend() const noexcept {
        return end();
    }

    reference operator[](size_type pos) noexcept {
        return core_.data()[pos];
    }

    const_reference operator[](size_type pos) const noexcept {
        return core_.data()[pos];
    }

    reference at(size_type pos) {
        if (pos >= core_.size()) {
            throw std::out_of_range("BasicString::at");
        }
        return core_.data()[pos];
    }

    const_reference at(size_type pos) const {
        if (pos >= core_.size()) {
            throw std::out_of_range("BasicString::at");
        }
        return core_.data()[pos];
    }

    reference front() noexcept {
        return *core_.data();
    }

    const_reference front() const noexcept {
        return *core_.data();
    }

    reference back() noexcept {
        return core_.data()[core_.size() - 1];
    }

    const_reference back() const noexcept {
        return core_.data()[core_.size() - 1];
    }

    void clear() noexcept {
        core_.clear();
    }

    void reserve(size_type new_cap = 0) {
        core_.reserve(new_cap);
    }

    void resize(size_type count) {
        core_.resize(count);
    }

    void resize(size_type count, CharT ch) {
        core_.resize(count, ch);
    }

    void shrink_to_fit() {
        core_.shrink_to_fit();
    }

    void push_back(CharT ch) {
        core_.push_back(ch);
    }

    void pop_back() {
        core_.pop_back();
    }

    BasicString& assign(const CharT* s, size_type n) {
        core_.assign(s, n);
        return *this;
    }

    BasicString& assign(std::basic_string_view<CharT, Traits> sv) {
        core_.assign(sv);
        return *this;
    }

    BasicString& assign(size_type count, CharT ch) {
        core_.assign(count, ch);
        return *this;
    }

    BasicString& append(const BasicString& str) {
        core_.append(str.data(), str.size());
        return *this;
    }

    BasicString& append(const CharT* s) {
        core_.append(s, traits_type::length(s));
        return *this;
    }

    BasicString& append(const CharT* s, size_type n) {
        core_.append(s, n);
        return *this;
    }

    BasicString& append(size_type count, CharT ch) {
        core_.append(count, ch);
        return *this;
    }

    BasicString& append(std::basic_string_view<CharT, Traits> sv) {
        core_.append(sv);
        return *this;
    }

    BasicString& operator+=(const BasicString& str) {
        return append(str);
    }

    BasicString& operator+=(const CharT* s) {
        return append(s);
    }

    BasicString& operator+=(CharT ch) {
        push_back(ch);
        return *this;
    }

    BasicString& operator+=(std::basic_string_view<CharT, Traits> sv) {
        return append(sv);
    }

    void swap(BasicString& other) noexcept(noexcept(core_.swap(other.core_))) {
        core_.swap(other.core_);
    }

    allocator_type get_allocator() const noexcept(noexcept(core_.get_allocator())) {
        return core_.get_allocator();
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_BASIC_STRING_HPP
