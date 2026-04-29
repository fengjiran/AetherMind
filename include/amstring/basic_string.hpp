// basic_string.hpp - Public API for amstring
// Part of AetherMind project, licensed under MIT License.
// See LICENSE.txt for details.
// SPDX-License-Identifier: MIT

#ifndef AETHERMIND_AMSTRING_BASIC_STRING_HPP
#define AETHERMIND_AMSTRING_BASIC_STRING_HPP

#include "core.hpp"
#include "growth_policy.hpp"
#include "string_fwd.hpp"

#include <algorithm>
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

    BasicString& insert(size_type pos, const BasicString& str) {
        return insert(pos, str.data(), str.size());
    }

    BasicString& insert(size_type pos, const BasicString& str, size_type subpos, size_type subcount = npos) {
        CheckPosition(subpos, str.size(), "BasicString::insert");
        return insert(pos, str.data() + subpos, ClampCount(str.size(), subpos, subcount));
    }

    BasicString& insert(size_type pos, const CharT* s) {
        return insert(pos, s, traits_type::length(s));
    }

    BasicString& insert(size_type pos, const CharT* s, size_type count) {
        CheckPosition(pos, size(), "BasicString::insert");
        return ReplaceRange(pos, 0, s, count);
    }

    BasicString& insert(size_type pos, size_type count, CharT ch) {
        CheckPosition(pos, size(), "BasicString::insert");
        BasicString tmp(count, ch);
        return ReplaceRange(pos, 0, tmp.data(), tmp.size());
    }

    BasicString& insert(size_type pos, std::basic_string_view<CharT, Traits> sv) {
        return insert(pos, sv.data(), sv.size());
    }

    BasicString& insert(size_type pos, std::basic_string_view<CharT, Traits> sv, size_type subpos, size_type subcount = npos) {
        CheckPosition(subpos, sv.size(), "BasicString::insert");
        return insert(pos, sv.data() + subpos, ClampCount(sv.size(), subpos, subcount));
    }

    iterator insert(const_iterator pos, CharT ch) {
        const size_type index = IndexFromIter(pos);
        insert(index, 1, ch);
        return begin() + index;
    }

    iterator insert(const_iterator pos, size_type count, CharT ch) {
        const size_type index = IndexFromIter(pos);
        insert(index, count, ch);
        return begin() + index;
    }

    BasicString& erase(size_type pos = 0, size_type count = npos) {
        CheckPosition(pos, size(), "BasicString::erase");
        return ReplaceRange(pos, count, nullptr, 0);
    }

    iterator erase(const_iterator pos) {
        const size_type index = IndexFromIter(pos);
        erase(index, 1);
        return begin() + index;
    }

    iterator erase(const_iterator first, const_iterator last) {
        const size_type first_index = IndexFromIter(first);
        const size_type last_index = IndexFromIter(last);
        erase(first_index, last_index - first_index);
        return begin() + first_index;
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

    BasicString& replace(size_type pos, size_type count, const BasicString& str) {
        return replace(pos, count, str.data(), str.size());
    }

    BasicString& replace(size_type pos, size_type count, const BasicString& str,
                         size_type subpos, size_type subcount = npos) {
        CheckPosition(subpos, str.size(), "BasicString::replace");
        return replace(pos, count, str.data() + subpos,
                       ClampCount(str.size(), subpos, subcount));
    }

    BasicString& replace(size_type pos, size_type count, const CharT* s) {
        return replace(pos, count, s, traits_type::length(s));
    }

    BasicString& replace(size_type pos, size_type count, const CharT* s, size_type count2) {
        return ReplaceRange(pos, count, s, count2);
    }

    BasicString& replace(size_type pos, size_type count, size_type count2, CharT ch) {
        BasicString tmp(count2, ch);
        return ReplaceRange(pos, count, tmp.data(), tmp.size());
    }

    BasicString& replace(size_type pos, size_type count, std::basic_string_view<CharT, Traits> sv) {
        return replace(pos, count, sv.data(), sv.size());
    }

    BasicString& replace(size_type pos, size_type count, std::basic_string_view<CharT, Traits> sv, size_type subpos, size_type subcount = npos) {
        CheckPosition(subpos, sv.size(), "BasicString::replace");
        return replace(pos, count, sv.data() + subpos, ClampCount(sv.size(), subpos, subcount));
    }

    BasicString substr(size_type pos = 0, size_type count = npos) const {
        CheckPosition(pos, size(), "BasicString::substr");
        return BasicString(data() + pos, ClampCount(size(), pos, count));
    }

    size_type find(const BasicString& str, size_type pos = 0) const noexcept {
        return find(str.data(), pos, str.size());
    }

    size_type find(const CharT* s, size_type pos, size_type count) const noexcept {
        return FindRange(data(), size(), s, pos, count);
    }

    size_type find(const CharT* s, size_type pos = 0) const noexcept {
        return find(s, pos, traits_type::length(s));
    }

    size_type find(CharT ch, size_type pos = 0) const noexcept {
        if (pos >= size()) {
            return npos;
        }
        const CharT* found = traits_type::find(data() + pos, size() - pos, ch);
        return found == nullptr ? npos : static_cast<size_type>(found - data());
    }

    size_type find(std::basic_string_view<CharT, Traits> sv, size_type pos = 0) const noexcept {
        return find(sv.data(), pos, sv.size());
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

private:
    using LayoutPolicy = DefaultLayoutPolicy<CharT>::type;
    using GrowthPolicy = DefaultGrowthPolicy;
    using CoreType = BasicStringCore<CharT, Traits, Allocator>;

    CoreType core_;

    static size_type ClampCount(size_type total, size_type pos, size_type count) noexcept {
        const size_type remaining = total - pos;
        return count == npos ? remaining : std::min(count, remaining);
    }

    static void CheckPosition(size_type pos, size_type total, const char* operation) {
        if (pos > total) {
            throw std::out_of_range(operation);
        }
    }

    size_type IndexFromIter(const_iterator it) const noexcept {
        return static_cast<size_type>(it - cbegin());
    }

    BasicString& ReplaceRange(size_type pos, size_type count, const CharT* src, size_type src_count) {
        CheckPosition(pos, size(), "BasicString::replace");
        const size_type erased = ClampCount(size(), pos, count);
        core_.replace_range(pos, erased, src, src_count);
        return *this;
    }

    static size_type FindRange(const CharT* haystack, size_type haystack_size, const CharT* needle, size_type pos, size_type needle_size) noexcept {
        if (needle_size == 0) {
            return pos <= haystack_size ? pos : npos;
        }
        if (pos >= haystack_size || needle_size > haystack_size - pos) {
            return npos;
        }

        const CharT first = needle[0];
        const size_type last_start = haystack_size - needle_size;
        for (size_type current = pos; current <= last_start;) {
            const CharT* found = traits_type::find(haystack + current, haystack_size - current - needle_size + 1, first);
            if (found == nullptr) {
                return npos;
            }

            current = static_cast<size_type>(found - haystack);
            if (traits_type::compare(haystack + current, needle, needle_size) == 0) {
                return current;
            }
            ++current;
        }
        return npos;
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_BASIC_STRING_HPP
