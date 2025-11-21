//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"
#include "error.h"

#include <cstring>
#include <utility>

namespace aethermind {

String::String(const_pointer other, size_type size) {
    if (other == nullptr) {
        if (size > 0) {
            AETHERMIND_THROW(LogicError) << "construction from null is not valid";
            AETHERMIND_UNREACHABLE();
        }
    } else {
        size = size > traits_type::length(other) ? traits_type::length(other) : size;
        Construct(other, other + size);
    }
}

String::String(const_pointer other) {
    if (other == nullptr) {
        AETHERMIND_THROW(LogicError) << "construction from null is not valid";
        AETHERMIND_UNREACHABLE();
    }

    Construct(other, other + traits_type::length(other));
}

String::String(size_type size, value_type c) {
    Construct(size, c);
}

String::String(const String& other) : size_(other.size_), impl_(other.impl_) {
    if (other.IsLocal()) {
        InitLocalBuffer();
        std::memcpy(local_buffer_, other.local_buffer_, other.size_);
    } else {
        capacity_ = other.capacity_;
    }
}

String::String(String&& other) noexcept : size_(other.size_), impl_(std::move(other.impl_)) {
    if (IsLocal()) {
        InitLocalBuffer();
        std::memcpy(local_buffer_, other.local_buffer_, size_);
    } else {
        capacity_ = other.capacity_;
        other.capacity_ = 0;
    }
    other.size_ = 0;
}

String::String(const String& other, size_type pos) {
    const_pointer start = other.data() + other.CheckPos(pos);
    Construct(start, start + other.Limit(pos, npos));
}

String::String(const String& other, size_type pos, size_type n) {
    const_pointer start = other.data() + other.CheckPos(pos);
    Construct(start, start + other.Limit(pos, n));
}

String::size_type String::max_size() noexcept {
    return (std::min<size_type>(allocator_traits::max_size(allocator_type()),
                                std::numeric_limits<size_type>::max()) -
            1) /
           2;
}

String& String::operator=(const String& other) {
    String(other).swap(*this);
    return *this;
}

String& String::operator=(String&& other) noexcept {
    String(std::move(other)).swap(*this);
    return *this;
}

String& String::operator=(const std::string& other) {
    String(other).swap(*this);
    return *this;
}

String& String::operator=(const char* other) {
    String(other).swap(*this);
    return *this;
}

String::pointer String::data() noexcept {
    return IsLocal() ? local_buffer_ : impl_->data();
}

String::const_pointer String::data() const noexcept {
    return IsLocal() ? local_buffer_ : impl_->data();
}

String::const_pointer String::c_str() const noexcept {
    return data();
}


// String::iterator String::begin() noexcept {
//     return data();
// }

String::iterator String::begin() noexcept {
    return iterator(*this, data());
}

// String::const_iterator String::begin() const noexcept {
//     return data();
// }

String::const_iterator String::begin() const noexcept {
    return const_iterator(*const_cast<String*>(this), data());
}

String::iterator String::end() noexcept {
    return iterator(*this, data() + size());
}

String::const_iterator String::end() const noexcept {
    return const_iterator(*const_cast<String*>(this), data() + size());
}

bool String::defined() const noexcept {
    return impl_;
}

bool String::IsLocal() const noexcept {
    return !defined();
}

String::size_type String::size() const noexcept {
    return size_;
}

String::size_type String::length() const noexcept {
    return size_;
}

String::size_type String::capacity() const noexcept {
    return IsLocal() ? static_cast<size_type>(local_capacity_) : capacity_;
}

bool String::empty() const noexcept {
    return size() == 0;
}

void String::clear() noexcept {
    size_ = 0;
    impl_.reset();
}

uint32_t String::use_count() const noexcept {
    return IsLocal() ? 1 : impl_.use_count();
}

bool String::unique() const noexcept {
    return use_count() == 1;
}

StringImpl* String::GetImplPtrUnsafe() const noexcept {
    return impl_.get();
}

StringImpl* String::ReleaseImplUnsafe() {
    return impl_.release();
}

const ObjectPtr<StringImpl>& String::GetObjectPtr() const {
    return impl_;
}

void String::InitLocalBuffer() noexcept {
    std::memset(local_buffer_, '\0', local_capacity_ + 1);
}

void String::push_back(value_type c) {
    append(1, c);
}

void String::pop_back() noexcept {
    CHECK(!empty());
    erase(size() - 1, 1);
}

String::const_reference String::operator[](size_type i) const noexcept {
    return data()[i];
}

String::CharProxy String::operator[](size_type i) noexcept {
    return {*this, i};
}

String::const_reference String::at(size_type i) const {
    if (i < size()) {
        return data()[i];
    }
    AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    AETHERMIND_UNREACHABLE();
}

String::CharProxy String::at(size_type i) {
    if (i < size()) {
        return {*this, i};
    }
    AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    AETHERMIND_UNREACHABLE();
}

String::CharProxy String::front() noexcept {
    CHECK(!empty());
    return {*this, 0};
}

String::const_reference String::front() const noexcept {
    CHECK(!empty());
    return operator[](0);
}

String::CharProxy String::back() noexcept {
    CHECK(!empty());
    return {*this, size() - 1};
}

String::const_reference String::back() const noexcept {
    CHECK(!empty());
    return operator[](size() - 1);
}

String String::substr(size_type pos, size_type n) const {
    return {*this, pos, n};
}

void String::resize(size_type n, value_type c) {
    if (const size_type sz = size(); n > sz) {
        append(n - sz, c);
    } else {
        size_ = n;
    }
}

void String::reserve(size_type n) {
    if (n > capacity()) {
        SwitchContainer(n);
    }
}

void String::shrink_to_fit() noexcept {
    if (!IsLocal()) {
        if (const size_type sz = size(); sz <= static_cast<size_type>(local_capacity_)) {
            InitLocalBuffer();
            std::memcpy(local_buffer_, data(), sz + 1);
            impl_.reset();
        } else {
            SwitchContainer(sz);
        }
    }
}

String& String::replace_aux(size_type pos, size_type n1, size_type n2) {
    if (n2 > max_size() - (size() - n1)) {
        AETHERMIND_THROW(out_of_range) << "The bytes to be allocated exceed the max_size()!";
    }

    pos = CheckPos(pos);
    n1 = Limit(pos, n1);
    if (n1 == 0 && n2 == 0) {
        return *this;
    }

    const auto delta = static_cast<int64_t>(n2 - n1);
    const size_type remain = size() - pos - n1;
    COW(delta);
    if (delta > 0) {
        pointer src = data() + size_;
        pointer dst = src + delta;
        for (size_type i = 0; i < remain; ++i) {
            *--dst = *--src;
        }
    } else if (delta < 0) {
        pointer src = data() + pos + n1;
        pointer dst = src + delta;
        for (size_type i = 0; i < remain; ++i) {
            *dst++ = *src++;
        }
    }

    size_ += delta;
    if (!IsLocal()) {// shrink to local buffer
        if (size() <= static_cast<size_type>(local_capacity_)) {
            InitLocalBuffer();
            std::memcpy(local_buffer_, data(), size() + 1);
            impl_.reset();
        }
    }
    return *this;
}

String& String::erase(size_type pos, size_type n) {
    pos = CheckPos(pos);
    if (pos == size()) {
        return *this;
    }
    return replace_aux(pos, Limit(pos, n), 0);
}

String::iterator String::erase(const_iterator position) {
    CHECK(position >= begin() && position < end()) << "erase position out of bounds";
    const size_type pos = position - begin();
    erase(pos, 1);
    return iterator(*this, data() + pos);
}

String::iterator String::erase(const_iterator first, const_iterator last) {
    CHECK(first >= begin() && first <= last && last <= end()) << "erase position out of bounds";
    const size_type pos = first - begin();
    const size_type n = last - first;
    erase(pos, n);
    return iterator(*this, data() + pos);
}

String& String::append(const_pointer src, size_type n) {
    return replace(size(), 0, src, n);
}

String& String::append(const_pointer src) {
    return replace(size(), 0, src, traits_type::length(src));
}

String& String::append(const String& str) {
    return replace(size(), 0, str);
}

String& String::append(const String& str, size_type pos, size_type n) {
    return replace(size(), 0, str, pos, n);
}

String& String::append(size_type n, value_type c) {
    return replace(size(), 0, n, c);
}

String& String::append(std::initializer_list<value_type> l) {
    return replace(size(), 0, l);
}

String& String::operator+=(const String& str) {
    return append(str);
}

String& String::operator+=(const_pointer str) {
    return append(str);
}

String& String::operator+=(value_type c) {
    return append(1, c);
}

String& String::operator+=(std::initializer_list<value_type> l) {
    return append(l);
}

String& String::replace(size_type pos, size_type n1, const_pointer str, size_type n2) {
    if (n1 == n2 && compare(pos, n1, str, n2) == 0) {
        return *this;
    }

    replace_aux(pos, n1, n2);
    std::memcpy(data() + pos, str, n2);
    return *this;
}

String& String::replace(size_type pos, size_type n1, const_pointer src) {
    return replace(pos, n1, src, traits_type::length(src));
}

String& String::replace(size_type pos, size_type n, const String& src) {
    return replace(pos, n, src.data(), src.size());
}

String& String::replace(size_type pos1, size_type n1, const String& src, size_type pos2, size_type n2) {
    return replace(pos1, n1, src.data() + src.CheckPos(pos2), src.Limit(pos2, n2));
}

String& String::replace(size_type pos, size_type n1, size_type n2, value_type c) {
    replace_aux(pos, n1, n2);
    std::memset(data() + pos, c, n2);
    return *this;
}

String& String::replace(const_iterator first, const_iterator last, const_pointer src, size_type n) {
    CHECK(first >= begin() && first <= last && last <= end());
    return replace(first - begin(), last - first, src, n);
}

String& String::replace(const_iterator first, const_iterator last, const String& src) {
    return replace(first, last, src.data(), src.size());
}

String& String::replace(const_iterator first, const_iterator last, const_pointer src) {
    return replace(first, last, src, traits_type::length(src));
}

String& String::replace(const_iterator first, const_iterator last, size_type n, value_type c) {
    CHECK(first >= begin() && first <= last && last <= end());
    return replace(first - begin(), last - first, n, c);
}

String& String::replace(const_iterator first, const_iterator last, pointer p1, pointer p2) {
    CHECK(first >= begin() && first <= last && last <= end());
    return replace(first - begin(), last - first, p1, p2 - p1);
}

String& String::replace(const_iterator first, const_iterator last, const_pointer p1, const_pointer p2) {
    CHECK(first >= begin() && first <= last && last <= end());
    return replace(first - begin(), last - first, p1, p2 - p1);
}

String& String::replace(const_iterator first, const_iterator last, std::initializer_list<value_type> l) {
    return replace(first, last, l.begin(), l.size());
}

String::iterator String::insert(const_iterator p, size_type n, value_type c) {
    CHECK(p >= begin() && p <= end());
    const size_type pos = p - begin();
    replace(p, p, n, c);
    return iterator(*this, data() + pos);
}

String::iterator String::insert(const_iterator p, std::initializer_list<char> l) {
    return insert(p, l.begin(), l.end());
}

String::iterator String::insert(const_iterator p, value_type c) {
    CHECK(p >= begin() && p <= end());
    const size_type pos = p - begin();
    insert(pos, 1, c);
    return iterator(*this, data() + pos);
}

String& String::insert(size_type pos, const String& other) {
    return replace(pos, 0, other);
}

String& String::insert(size_type pos1, const String& other, size_type pos2, size_type n) {
    return replace(pos1, 0, other, other.CheckPos(pos2), other.Limit(pos2, n));
}

String& String::insert(size_type pos, const_pointer str, size_type n) {
    return replace(pos, 0, str, n);
}

String& String::insert(size_type pos, const_pointer str) {
    return replace(pos, 0, str, traits_type::length(str));
}

String& String::insert(size_type pos, size_type n, value_type c) {
    return replace(pos, 0, n, c);
}

String::size_type String::find(const_pointer s, size_type pos, size_type n) const noexcept {
    const size_type sz = size();
    if (n == 0) {
        return pos <= sz ? pos : npos;
    }

    if (pos >= sz || sz - pos < n) {
        return npos;
    }

    const value_type elem0 = s[0];
    while (pos <= sz - n) {
        size_type i = pos;
        bool found = false;
        while (i <= sz - n) {
            if (traits_type::eq(data()[i], elem0)) {
                found = true;
                break;
            }
            ++i;
        }

        if (!found) {
            return npos;
        }

        if (compare(i, n, s, n) == 0) {
            return i;
        }

        pos = i + 1;
    }

    return npos;
}

String::size_type String::find_kmp(const_pointer s, size_type pos, size_type n) const noexcept {
    const size_type sz = size();
    if (n == 0) {
        return pos <= sz ? pos : npos;
    }

    if (pos >= sz) {
        return npos;
    }

    const value_type elem0 = s[0];
    const const_pointer start = data();
    const_pointer first = start + pos;
    const const_pointer last = start + sz;
    size_type len = sz - pos;

    while (len >= n) {
        // find the first occurrence of elem0
        const_pointer first_occur = nullptr;
        for (size_type i = 0; i < len - n + 1; ++i) {
            if (*(first + i) == elem0) {
                first_occur = first + i;
                break;
            }
        }

        if (first_occur == nullptr) {
            return npos;
        }

        first = first_occur;
        if (compare(first - start, n, s, n) == 0) {
            return first - start;
        }
        len = last - ++first;
    }
    return npos;
}

String::size_type String::find(const String& str, size_type pos) const noexcept {
    return find(str.data(), pos, str.size());
}

String::size_type String::find(const_pointer str, size_type pos) const noexcept {
    return find(str, pos, traits_type::length(str));
}

String::size_type String::find(value_type c, size_type pos) const noexcept {
    const size_type sz = size();
    if (pos >= sz) {
        return npos;
    }

    for (size_type i = pos; i < sz; ++i) {
        if (*(data() + i) == c) {
            return i;
        }
    }
    return npos;
}

String::size_type String::rfind(const_pointer s, size_type pos, size_type n) const noexcept {
    const size_type sz = size();
    if (n > sz) {
        return npos;
    }

    pos = std::min(pos, sz - n);
    do {
        if (compare(pos, n, s, n) == 0) {
            return pos;
        }
    } while (pos-- > 0);

    return npos;
}

String::size_type String::rfind(const String& str, size_type pos) const noexcept {
    return rfind(str.data(), pos, str.size());
}

String::size_type String::rfind(const_pointer str, size_type pos) const noexcept {
    return rfind(str, pos, traits_type::length(str));
}

String::size_type String::rfind(value_type c, size_type pos) const noexcept {
    if (empty()) {
        return npos;
    }

    size_type sz = size();
    if (--sz > pos) {
        sz = pos;
    }

    do {
        if (*(data() + sz) == c) {
            return sz;
        }
    } while (sz-- > 0);

    return npos;
}


String::size_type String::find_first_of(const_pointer s, size_type pos, size_type n) const noexcept {
    if (n == 0) {
        return npos;
    }

    const size_type sz = size();
    while (pos < sz) {
        for (size_type i = 0; i < n; ++i) {
            if (data()[pos] == s[i]) {
                return pos;
            }
        }
        ++pos;
    }
    return npos;
}

String::size_type String::find_first_of(const String& str, size_type pos) const noexcept {
    return find_first_of(str.data(), pos, str.size());
}

String::size_type String::find_first_of(const_pointer str, size_type pos) const noexcept {
    return find_first_of(str, pos, traits_type::length(str));
}

String::size_type String::find_first_of(value_type c, size_type pos) const noexcept {
    return find(c, pos);
}

String::size_type String::find_first_not_of(const_pointer s, size_type pos, size_type n) const noexcept {
    if (n == 0) {
        return npos;
    }

    const size_type sz = size();
    while (pos < sz) {
        bool found = false;
        for (size_type i = 0; i < n; ++i) {
            if (data()[pos] == s[i]) {
                found = true;
                break;
            }
        }

        if (!found) {
            return pos;
        }
        ++pos;
    }
    return npos;
}

String::size_type String::find_first_not_of(const String& str, size_type pos) const noexcept {
    return find_first_not_of(str.data(), pos, str.size());
}

String::size_type String::find_first_not_of(const_pointer str, size_type pos) const noexcept {
    return find_first_not_of(str, pos, traits_type::length(str));
}

String::size_type String::find_first_not_of(value_type c, size_type pos) const noexcept {
    const size_type sz = size();
    while (pos < sz) {
        if (!traits_type::eq(data()[pos], c)) {
            return pos;
        }
        ++pos;
    }
    return npos;
}

String::size_type String::find_last_of(const_pointer s, size_type pos, size_type n) const noexcept {
    size_type sz = size();
    if (n == 0 || sz == 0) {
        return npos;
    }

    if (--sz > pos) {
        sz = pos;
    }

    do {
        for (size_type i = 0; i < n; ++i) {
            if (data()[sz] == s[i]) {
                return sz;
            }
        }
    } while (sz-- > 0);

    return npos;
}

String::size_type String::find_last_of(const String& str, size_type pos) const noexcept {
    return find_last_of(str.data(), pos, str.size());
}

String::size_type String::find_last_of(const_pointer str, size_type pos) const noexcept {
    return find_last_of(str, pos, traits_type::length(str));
}

String::size_type String::find_last_of(value_type c, size_type pos) const noexcept {
    return rfind(c, pos);
}

String::size_type String::find_last_not_of(const_pointer s, size_type pos, size_type n) const noexcept {
    if (empty()) {
        return npos;
    }

    size_type sz = size();
    if (--sz > pos) {
        sz = pos;
    }

    do {
        bool found = false;
        for (size_type i = 0; i < n; ++i) {
            if (data()[sz] == s[i]) {
                found = true;
                break;
            }
        }

        if (!found) {
            return sz;
        }
    } while (sz-- > 0);

    return npos;
}

String::size_type String::find_last_not_of(const String& str, size_type pos) const noexcept {
    return find_last_not_of(str.data(), pos, str.size());
}

String::size_type String::find_last_not_of(const_pointer str, size_type pos) const noexcept {
    return find_last_not_of(str, pos, traits_type::length(str));
}

String::size_type String::find_last_not_of(value_type c, size_type pos) const noexcept {
    if (empty()) {
        return npos;
    }

    size_type sz = size();
    if (--sz > pos) {
        sz = pos;
    }

    do {
        if (data()[sz] != c) {
            return sz;
        }
    } while (sz-- > 0);

    return npos;
}

bool String::starts_with(const String& str) const noexcept {
    return compare(0, str.size(), str) == 0;
}

bool String::starts_with(const_pointer str) const noexcept {
    return compare(0, traits_type::length(str), str) == 0;
}

bool String::starts_with(value_type c) const noexcept {
    return empty() ? false : traits_type::eq(front(), c);
}

bool String::ends_with(const String& str) const noexcept {
    if (str.size() > size()) {
        return false;
    }

    return compare(size() - str.size(), str.size(), str) == 0;
}

bool String::ends_with(const_pointer str) const noexcept {
    const size_type len = traits_type::length(str);
    if (len > size()) {
        return false;
    }

    return compare(size() - len, len, str) == 0;
}

bool String::ends_with(value_type c) const noexcept {
    return empty() ? false : traits_type::eq(back(), c);
}

String::operator std::string() const {
    return {data(), size()};
}

String::operator const char*() const {
    return data();
}

void String::swap(String& other) noexcept {
    if (this == &other) {
        return;
    }

    if (IsLocal()) {
        if (other.IsLocal()) {
            if (!empty() && !other.empty()) {
                value_type tmp[local_capacity_ + 1];
                std::memcpy(tmp, other.local_buffer_, other.size() + 1);
                std::memcpy(other.local_buffer_, local_buffer_, size() + 1);
                std::memcpy(local_buffer_, tmp, other.size() + 1);
            } else if (!other.empty()) {
                InitLocalBuffer();
                std::memcpy(local_buffer_, other.local_buffer_, other.size() + 1);
            } else if (!empty()) {
                other.InitLocalBuffer();
                std::memcpy(other.local_buffer_, local_buffer_, size() + 1);
            }
        } else {
            const auto tmp_cap = other.capacity_;
            other.InitLocalBuffer();
            std::memcpy(other.local_buffer_, local_buffer_, size() + 1);
            capacity_ = tmp_cap;
        }
    } else {
        if (other.IsLocal()) {
            const auto tmp_cap = capacity_;
            InitLocalBuffer();
            std::memcpy(local_buffer_, other.local_buffer_, other.size() + 1);
            other.capacity_ = tmp_cap;
        } else {
            std::swap(capacity_, other.capacity_);
        }
    }

    std::swap(impl_, other.impl_);
    std::swap(size_, other.size_);
}

void String::Construct(size_type n, value_type c) {
    pointer dst = nullptr;
    if (n > static_cast<size_type>(local_capacity_)) {
        impl_ = StringImpl::Create(n);
        capacity_ = n;
        dst = impl_->data();
    } else {
        InitLocalBuffer();
        dst = local_buffer_;
    }
    std::memset(dst, c, n);
    size_ = n;
}

void String::SwitchContainer(size_type new_cap) {
    auto impl = StringImpl::Create(new_cap);
    std::memcpy(impl->data(), data(), size());
    impl_ = impl;
    capacity_ = new_cap;
}

void String::COW(int64_t delta) {
    if (delta > 0) {// expand
        const size_type new_size = static_cast<size_t>(delta) + size();
        if (IsLocal()) {
            if (new_size > capacity()) {
                const size_type new_cap = std::max(new_size, capacity() * kIncFactor);
                SwitchContainer(new_cap);
            }
        } else {
            if (unique()) {
                if (new_size > capacity()) {
                    const size_type new_cap = std::max(new_size, capacity() * kIncFactor);
                    SwitchContainer(new_cap);
                }
            } else {
                const size_type new_cap = new_size > capacity() ? std::max(new_size, capacity() * kIncFactor)
                                                                : capacity();
                SwitchContainer(new_cap);
            }
        }
    } else {// inplace or shrink
        if (!IsLocal() && !unique()) {
            SwitchContainer(capacity());
        }

        // size_type new_size = size() + delta;
        // if (!IsLocal()) {
        //     if (new_size <= static_cast<size_type>(local_capacity_)) {
        //         InitLocalBuffer();
        //         std::memcpy(local_buffer_, data(), new_size + 1);
        //         impl_.reset();
        //     } else {
        //         if (!unique()) {
        //             SwitchContainer(new_size);
        //             // SwitchContainer(capacity());
        //         }
        //     }
        // }
    }
}

String::size_type String::Limit(size_type pos, size_type limit) const noexcept {
    return size() - pos > limit ? limit : size() - pos;
}

String::size_type String::CheckPos(size_type pos) const {
    if (pos > size()) {
        AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    }
    return pos;
}

void String::CheckSize(size_type delta) const {
    if (delta > max_size() - size()) {
        AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    }
}

int String::MemoryCompare(const_pointer lhs, size_type lhs_cnt, const_pointer rhs, size_type rhs_cnt) {
    if (lhs == rhs && lhs_cnt == rhs_cnt) {
        return 0;
    }

    for (size_t i = 0; i < std::min(lhs_cnt, rhs_cnt); ++i) {
        if (lhs[i] < rhs[i]) {
            return -1;
        }

        if (lhs[i] > rhs[i]) {
            return 1;
        }
    }

    if (lhs_cnt < rhs_cnt) {
        return -1;
    }

    if (lhs_cnt > rhs_cnt) {
        return 1;
    }

    return 0;
}

int String::compare(const String& other) const {
    return MemoryCompare(data(), size(), other.data(), other.size());
}

int String::compare(size_type pos, size_type n, const String& other) const {
    return MemoryCompare(data() + CheckPos(pos), Limit(pos, n),
                         other.data(), other.size());
}

int String::compare(size_type pos1, size_type n1,
                    const String& other, size_type pos2, size_type n2) const {
    return MemoryCompare(data() + CheckPos(pos1), Limit(pos1, n1),
                         other.data() + other.CheckPos(pos2), other.Limit(pos2, n2));
}

int String::compare(const std::string& other) const {
    return MemoryCompare(data(), size(), other.data(), other.size());
}

int String::compare(size_type pos, size_type n, const std::string& other) const {
    return MemoryCompare(data() + CheckPos(pos), Limit(pos, n), other.data(), other.size());
}

int String::compare(size_type pos1, size_type n1, const std::string& other, size_type pos2, size_type n2) const {
    if (pos2 > other.size()) {
        AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    }

    n2 = n2 > other.size() - pos2 ? other.size() - pos2 : n2;
    return MemoryCompare(data() + CheckPos(pos1), Limit(pos1, n1),
                         other.data() + pos2, n2);
}


int String::compare(const_pointer other) const {
    return MemoryCompare(data(), size(), other, traits_type::length(other));
}

int String::compare(size_type pos, size_type n, const_pointer other) const {
    return MemoryCompare(data() + CheckPos(pos), Limit(pos, n),
                         other, traits_type::length(other));
}

int String::compare(size_type pos, size_type n1, const_pointer other, size_type n2) const {
    // n2 = n2 > traits_type::length(other) ? traits_type::length(other) : n2;
    return MemoryCompare(data() + CheckPos(pos), Limit(pos, n1),
                         other, n2);
}

String String::Concat(const_pointer lhs, size_t lhs_cnt, const_pointer rhs, size_t rhs_cnt) {
    String res(lhs, lhs_cnt);
    res.append(rhs, rhs_cnt);
    return res;
}

String operator+(const String& lhs, const String& rhs) {
    return String::Concat(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

String operator+(const String& lhs, const std::string& rhs) {
    return String::Concat(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

String operator+(const std::string& lhs, const String& rhs) {
    return String::Concat(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

String operator+(const String& lhs, String::const_pointer rhs) {
    return String::Concat(lhs.data(), lhs.size(), rhs, std::strlen(rhs));
}

String operator+(String::const_pointer lhs, const String& rhs) {
    return String::Concat(lhs, std::strlen(lhs), rhs.data(), rhs.size());
}

String operator+(const String& lhs, String::value_type rhs) {
    String res(lhs);
    res += rhs;
    return res;
}

String operator+(String::value_type lhs, const String& rhs) {
    return rhs + lhs;
}

}// namespace aethermind