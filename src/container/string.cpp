//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"
#include "error.h"

#include <cstring>
#include <utility>

namespace aethermind_test {

StringImpl::StringImpl() : data_(nullptr), size_(0), capacity_(0) {}

size_t StringImpl::size() const noexcept {
    return size_;
}

size_t StringImpl::capacity() const noexcept {
    return capacity_;
}

const char* StringImpl::data() const noexcept {
    return data_;
}

ObjectPtr<StringImpl> StringImpl::Create(size_t cap) {
    auto impl = make_array_object<StringImpl, char>(cap + 1);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(StringImpl);
    impl->size_ = 0;
    impl->capacity_ = cap;
    std::memset(impl->data_, '\0', cap + 1);
    return impl;
}

String::String() : size_(0) {
    InitLocalBuffer();
}

String::String(const char* other, size_t size) {
    if (other == nullptr) {
        if (size > 0) {
            AETHERMIND_THROW(LogicError) << "construction from null is not valid";
        }
    } else {
        impl_ = StringImpl::Create(size);
        std::memcpy(impl_->data_, other, size);
        impl_->size_ = size;
    }
}

String::String(const char* other) {
    if (other == nullptr) {
        AETHERMIND_THROW(LogicError) << "construction from null is not valid";
    }

    const auto size = std::strlen(other);
    impl_ = StringImpl::Create(size);
    std::memcpy(impl_->data_, other, size);
    impl_->size_ = size;
}

String::String(size_t size, char c) : impl_(StringImpl::Create(size)) {
    std::memset(impl_->data_, c, size);
    impl_->size_ = size;
}

String::String(std::initializer_list<char> list) : String(list.begin(), list.end()) {}

String::String(const std::string& other) : String(other.data(), other.size()) {}

String::String(std::string&& other) : String(other.data(), other.size()) {
    other.clear();
}

String::String(std::string_view other) : String(other.data(), other.size()) {}

String::String(ObjectPtr<StringImpl> impl) : impl_(std::move(impl)) {}

void String::swap(String& other) noexcept {
    std::swap(impl_, other.impl_);
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

const char* String::data() const noexcept {
    return impl_->data();
}

const char* String::c_str() const noexcept {
    return data();
}

bool String::defined() const noexcept {
    return impl_;
}

size_t String::size() const noexcept {
    return impl_->size();
}

bool String::empty() const noexcept {
    return size() == 0;
}

char String::operator[](size_t i) const noexcept {
    return data()[i];
}

char String::at(size_t i) const {
    if (i < size()) {
        return data()[i];
    }
    AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    AETHERMIND_UNREACHABLE();
}

uint32_t String::use_count() const noexcept {
    return impl_.use_count();
}

bool String::unique() const noexcept {
    return impl_.unique();
}

StringImpl* String::GetImplPtrUnsafe() const noexcept {
    return impl_.get();
}

const ObjectPtr<StringImpl>& String::GetObjectPtr() const {
    return impl_;
}

StringImpl* String::ReleaseImplUnsafe() {
    return impl_.release();
}

String::operator std::string() const {
    return {data(), size()};
}

String::operator const char*() const {
    return data();
}

void String::InitLocalBuffer() noexcept {
    std::memset(local_buffer_, '\0', local_capacity_ + 1);
}

bool String::IsLocal() const noexcept {
    return !defined();
}


int String::Compare(const String& other) const {
    return MemoryCompare(data(), size(), other.data(), other.size());
}

int String::Compare(const std::string& other) const {
    return MemoryCompare(data(), size(), other.data(), other.size());
}

int String::Compare(const char* other) const {
    return MemoryCompare(data(), size(), other, std::strlen(other));
}

int String::MemoryCompare(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
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

bool String::MemoryEqual(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
    return MemoryCompare(lhs, lhs_cnt, rhs, rhs_cnt) == 0;
}

String String::Concat(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
    auto ptr = make_array_object<StringImpl, char>(lhs_cnt + rhs_cnt + 1);
    char* dst = reinterpret_cast<char*>(ptr.get()) + sizeof(StringImpl);
    ptr->data_ = dst;
    ptr->size_ = lhs_cnt + rhs_cnt;
    std::memcpy(dst, lhs, lhs_cnt);
    std::memcpy(dst + lhs_cnt, rhs, rhs_cnt);
    dst[lhs_cnt + rhs_cnt] = '\0';
    return String(std::move(ptr));
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

String operator+(const String& lhs, const char* rhs) {
    return String::Concat(lhs.data(), lhs.size(), rhs, std::strlen(rhs));
}

String operator+(const char* lhs, const String& rhs) {
    return String::Concat(lhs, std::strlen(lhs), rhs.data(), rhs.size());
}

std::ostream& operator<<(std::ostream& os, const String& str) {
    os.write(str.data(), str.size());
    return os;
}


}// namespace aethermind_test

namespace aethermind {

String::String(const char* other, size_type size) {
    if (other == nullptr) {
        if (size > 0) {
            AETHERMIND_THROW(LogicError) << "construction from null is not valid";
            AETHERMIND_UNREACHABLE();
        }
    } else {
        Construct(other, other + size);
    }
}

String::String(const char* other) {
    if (other == nullptr) {
        AETHERMIND_THROW(LogicError) << "construction from null is not valid";
        AETHERMIND_UNREACHABLE();
    }

    Construct(other, other + std::strlen(other));
}

String::String(size_type size, char c) {
    Construct(size, c);
}

String::String(const String& other) : size_(other.size_), impl_(other.impl_) {
    if (other.IsLocal()) {
        InitLocalBuffer();
        std::memcpy(local_buffer_, other.local_buffer_, size_);
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

String::value_type String::at(size_type i) const {
    if (i < size()) {
        return data()[i];
    }
    AETHERMIND_THROW(out_of_range) << "String index out of bounds";
    AETHERMIND_UNREACHABLE();
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
                char tmp[local_capacity_ + 1];
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

void String::Construct(size_type n, char c) {
    char* dst = nullptr;
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

bool String::MemoryEqual(const_pointer lhs, size_type lhs_cnt, const_pointer rhs, size_type rhs_cnt) {
    return MemoryCompare(lhs, lhs_cnt, rhs, rhs_cnt) == 0;
}

int String::Compare(const String& other) const {
    return MemoryCompare(data(), size(), other.data(), other.size());
}

int String::Compare(const std::string& other) const {
    return MemoryCompare(data(), size(), other.data(), other.size());
}

int String::Compare(const_pointer other) const {
    return MemoryCompare(data(), size(), other, std::strlen(other));
}

String String::Concat(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
    String res(lhs_cnt + rhs_cnt, '\0');
    std::memcpy(res.data(), lhs, lhs_cnt);
    std::memcpy(res.data() + lhs_cnt, rhs, rhs_cnt);
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

String operator+(const String& lhs, const char* rhs) {
    return String::Concat(lhs.data(), lhs.size(), rhs, std::strlen(rhs));
}

String operator+(const char* lhs, const String& rhs) {
    return String::Concat(lhs, std::strlen(lhs), rhs.data(), rhs.size());
}

}// namespace aethermind