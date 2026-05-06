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
#include <utility>

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

    AM_NODISCARD size_type size() const noexcept {
        return core_.size();
    }

    AM_NODISCARD size_type length() const noexcept {
        return core_.size();
    }

    AM_NODISCARD size_type capacity() const noexcept {
        return core_.capacity();
    }

    AM_NODISCARD bool empty() const noexcept {
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
        const_pointer source = str.data();
        const size_type source_size = str.size();
        return ReplaceRange(pos, 0, source, source_size, "BasicString::insert");
    }

    BasicString& insert(size_type pos, const BasicString& str, size_type subpos, size_type subcount = npos) {
        const_pointer source = str.data();
        const size_type source_size = str.size();
        const size_type normalized_count = CheckedClampCount(source_size, subpos, subcount, "BasicString::insert");
        return ReplaceRange(pos, 0, source + subpos, normalized_count, "BasicString::insert");
    }

    BasicString& insert(size_type pos, const CharT* s) {
        const size_type source_size = traits_type::length(s);
        return ReplaceRange(pos, 0, s, source_size, "BasicString::insert");
    }

    BasicString& insert(size_type pos, const CharT* s, size_type count) {
        return ReplaceRange(pos, 0, s, count, "BasicString::insert");
    }

    BasicString& insert(size_type pos, size_type count, CharT ch) {
        return ReplaceFillRange(pos, 0, count, ch, "BasicString::insert");
    }

    BasicString& insert(size_type pos, std::basic_string_view<CharT, Traits> sv) {
        return ReplaceRange(pos, 0, sv.data(), sv.size(), "BasicString::insert");
    }

    BasicString& insert(size_type pos, std::basic_string_view<CharT, Traits> sv, size_type subpos, size_type subcount = npos) {
        const size_type source_size = sv.size();
        const size_type normalized_count = CheckedClampCount(source_size, subpos, subcount, "BasicString::insert");
        return ReplaceRange(pos, 0, sv.data() + subpos, normalized_count, "BasicString::insert");
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
        return ReplaceRange(pos, count, nullptr, 0, "BasicString::erase");
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
        const_pointer source = str.data();
        const size_type source_size = str.size();
        return ReplaceRange(pos, count, source, source_size, "BasicString::replace");
    }

    BasicString& replace(size_type pos, size_type count, const BasicString& str,
                         size_type subpos, size_type subcount = npos) {
        const_pointer source = str.data();
        const size_type source_size = str.size();
        const size_type normalized_count = CheckedClampCount(source_size, subpos, subcount, "BasicString::replace");
        return ReplaceRange(pos, count, source + subpos, normalized_count, "BasicString::replace");
    }

    BasicString& replace(size_type pos, size_type count, const CharT* s) {
        const size_type source_size = traits_type::length(s);
        return ReplaceRange(pos, count, s, source_size, "BasicString::replace");
    }

    BasicString& replace(size_type pos, size_type count, const CharT* s, size_type count2) {
        return ReplaceRange(pos, count, s, count2, "BasicString::replace");
    }

    BasicString& replace(size_type pos, size_type count, size_type count2, CharT ch) {
        return ReplaceFillRange(pos, count, count2, ch, "BasicString::replace");
    }

    BasicString& replace(size_type pos, size_type count, std::basic_string_view<CharT, Traits> sv) {
        return ReplaceRange(pos, count, sv.data(), sv.size(), "BasicString::replace");
    }

    BasicString& replace(size_type pos, size_type count, std::basic_string_view<CharT, Traits> sv, size_type subpos, size_type subcount = npos) {
        const size_type source_size = sv.size();
        const size_type normalized_count = CheckedClampCount(source_size, subpos, subcount, "BasicString::replace");
        return ReplaceRange(pos, count, sv.data() + subpos, normalized_count, "BasicString::replace");
    }

    BasicString substr(size_type pos = 0, size_type count = npos) const {
        const_pointer d = core_.data();
        const size_type sz = core_.size();
        const size_type normalized_count = CheckedClampCount(sz, pos, count, "BasicString::substr");
        return BasicString(d + pos, normalized_count);
    }

    size_type find(const BasicString& str, size_type pos = 0) const noexcept {
        return find(str.data(), pos, str.size());
    }

    size_type find(const CharT* needle, size_type pos, size_type count) const noexcept {
        return FindRange(core_.data(), core_.size(), needle, pos, count);
    }

    size_type find(const CharT* needle, size_type pos = 0) const noexcept {
        return find(needle, pos, traits_type::length(needle));
    }

    size_type find(CharT ch, size_type pos = 0) const noexcept {
        const size_type sz = core_.size();
        if (pos >= sz) {
            return npos;
        }

        const_pointer d = core_.data();
        const_pointer found = traits_type::find(d + pos, sz - pos, ch);
        return found == nullptr ? npos : static_cast<size_type>(found - d);
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

    static size_type CheckedClampCount(size_type total, size_type pos, size_type count, const char* operation) {
        CheckPosition(pos, total, operation);
        return ClampCount(total, pos, count);
    }

    static void CheckPosition(size_type pos, size_type total, const char* operation) {
        if (pos > total) {
            throw std::out_of_range(operation);
        }
    }

    size_type IndexFromIter(const_iterator it) const noexcept {
        return static_cast<size_type>(it - cbegin());
    }

    BasicString& ReplaceRange(size_type pos, size_type count, const CharT* src, size_type src_count, const char* operation) {
        const size_type current_size = core_.size();
        const size_type erased = CheckedClampCount(current_size, pos, count, operation);
        core_.replace_range(pos, erased, src, src_count);
        return *this;
    }

    BasicString& ReplaceFillRange(size_type pos, size_type count, size_type fill_count, CharT ch, const char* operation) {
        const size_type current_size = core_.size();
        const size_type erased = CheckedClampCount(current_size, pos, count, operation);
        core_.replace_range(pos, erased, fill_count, ch);
        return *this;
    }

    static size_type FindRange(const CharT* haystack, size_type haystack_size,
                               const CharT* needle, size_type pos, size_type needle_size) noexcept {
        if (needle_size == 0) {
            return pos <= haystack_size ? pos : npos;
        }

        if (pos >= haystack_size || needle_size > haystack_size - pos) {
            return npos;
        }

        if (needle_size == 1) {
            const CharT* found = traits_type::find(haystack + pos, haystack_size - pos, needle[0]);
            return found == nullptr ? npos : static_cast<size_type>(found - haystack);
        }

        if (const size_type remaining = haystack_size - pos; needle_size < 8 || remaining < 128) {
            return FindRangeNaive(haystack, haystack_size, needle, pos, needle_size);
        }

        return FindRangeHybrid(haystack, haystack_size, needle, pos, needle_size);
    }

    static size_type FindRangeNaive(const CharT* haystack, size_type haystack_size,
                                    const CharT* needle, size_type pos, size_type needle_size) noexcept {
        AM_DCHECK(needle_size > 0);
        AM_DCHECK(pos < haystack_size);
        AM_DCHECK(needle_size <= haystack_size - pos);

        const CharT first = needle[0];
        const size_type last_start = haystack_size - needle_size;
        size_type cur = pos;
        while (cur <= last_start) {
            const CharT* found = traits_type::find(haystack + cur, haystack_size - cur - needle_size + 1, first);
            if (found == nullptr) {
                return npos;
            }

            cur = static_cast<size_type>(found - haystack);
            if (traits_type::compare(haystack + cur, needle, needle_size) == 0) {
                return cur;
            }
            ++cur;
        }
        return npos;
    }

    static size_type FindRangeHybrid(const CharT* haystack, size_type haystack_size,
                                     const CharT* needle, size_type pos, size_type needle_size) noexcept {
        AM_DCHECK(needle_size >= 2);
        AM_DCHECK(pos < haystack_size);
        AM_DCHECK(needle_size <= haystack_size - pos);

        constexpr size_type kMaxNaiveCandidates = 16;
        const CharT first = needle[0];
        const size_type last_start = haystack_size - needle_size;
        size_type failed_candidates = 0;
        size_type cur = pos;
        while (cur <= last_start) {
            const CharT* found = traits_type::find(haystack + cur, haystack_size - cur - needle_size + 1, first);
            if (found == nullptr) {
                return npos;
            }

            cur = static_cast<size_type>(found - haystack);
            if (traits_type::compare(haystack + cur, needle, needle_size) == 0) {
                return cur;
            }

            ++cur;
            ++failed_candidates;
            if (failed_candidates == kMaxNaiveCandidates && cur <= last_start) {
                return FindRangeTwoWay(haystack, haystack_size, needle, cur, needle_size);
            }
        }
        return npos;
    }

    static std::pair<size_type, size_type> MaximalSuffix(const CharT* needle, size_type needle_size, bool reversed) noexcept {
        size_type suffix = 0;
        size_type candidate = 1;
        size_type offset = 0;
        size_type period = 1;

        while (candidate + offset < needle_size) {
            const CharT candidate_ch = needle[candidate + offset];
            const CharT suffix_ch = needle[suffix + offset];

            const bool candidate_is_smaller = reversed ? traits_type::lt(suffix_ch, candidate_ch)
                                                       : traits_type::lt(candidate_ch, suffix_ch);
            if (candidate_is_smaller) {
                candidate += offset + 1;
                offset = 0;
                period = candidate - suffix;
            } else if (traits_type::eq(candidate_ch, suffix_ch)) {
                if (offset + 1 == period) {
                    candidate += period;
                    offset = 0;
                } else {
                    ++offset;
                }
            } else {
                suffix = candidate;
                ++candidate;
                offset = 0;
                period = 1;
            }
        }

        return {suffix, period};
    }

    static std::pair<size_type, size_type> CriticalFactorization(const CharT* needle, size_type needle_size) noexcept {
        const auto forward = MaximalSuffix(needle, needle_size, false);
        const auto reverse = MaximalSuffix(needle, needle_size, true);
        return forward.first > reverse.first ? forward : reverse;
    }

    static bool HasPeriodAtCriticalPosition(const CharT* needle, size_type needle_size, size_type period, size_type critical_position) noexcept {
        return critical_position <= needle_size - period &&
               traits_type::compare(needle, needle + period, critical_position) == 0;
    }

    static size_type FindRangeTwoWay(const CharT* haystack, size_type haystack_size, const CharT* needle, size_type pos, size_type needle_size) noexcept {
        AM_DCHECK(needle_size >= 2);
        AM_DCHECK(pos < haystack_size);
        AM_DCHECK(needle_size <= haystack_size - pos);

        const auto factorization = CriticalFactorization(needle, needle_size);
        const size_type critical_position = factorization.first;
        const size_type period = factorization.second;
        const size_type last_start = haystack_size - needle_size;

        if (HasPeriodAtCriticalPosition(needle, needle_size, period, critical_position)) {
            size_type memory = 0;
            for (size_type current = pos; current <= last_start;) {
                size_type index = std::max(critical_position, memory);
                while (index < needle_size && traits_type::eq(needle[index], haystack[current + index])) {
                    ++index;
                }
                if (index < needle_size) {
                    current += index - critical_position + 1;
                    memory = 0;
                    continue;
                }

                index = critical_position;
                while (index > memory && traits_type::eq(needle[index - 1], haystack[current + index - 1])) {
                    --index;
                }
                if (index == memory) {
                    return current;
                }

                current += period;
                memory = needle_size - period;
            }
            return npos;
        }

        const size_type shift = std::max(critical_position, needle_size - critical_position) + 1;
        for (size_type current = pos; current <= last_start;) {
            size_type index = critical_position;
            while (index < needle_size && traits_type::eq(needle[index], haystack[current + index])) {
                ++index;
            }
            if (index < needle_size) {
                current += index - critical_position + 1;
                continue;
            }

            index = critical_position;
            while (index > 0 && traits_type::eq(needle[index - 1], haystack[current + index - 1])) {
                --index;
            }
            if (index == 0) {
                return current;
            }

            current += shift;
        }
        return npos;
    }
};

}// namespace aethermind

#endif// AETHERMIND_AMSTRING_BASIC_STRING_HPP
