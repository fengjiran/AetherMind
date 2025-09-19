//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"
#include "error.h"

#include <cstring>
#include <utility>

namespace aethermind {

StringImpl::StringImpl() : data_(nullptr), size_(0) {}

size_t StringImpl::size() const noexcept {
    return size_;
}

const char* StringImpl::data() const noexcept {
    return data_;
}

String::String(const char* other, size_t size)
    : impl_(make_array_object<StringImpl, char>(size + 1)) {
    char* dst = reinterpret_cast<char*>(impl_.get()) + sizeof(StringImpl);
    impl_->data_ = dst;
    impl_->size_ = size;
    std::memcpy(dst, other, size);
    dst[size] = '\0';
}

String::String(const char* other) : String(other, std::strlen(other)) {}

String::String(const std::string& other) : String(other.data(), other.size()) {}

String::String(std::string&& other) : String(other.data(), other.size()) {
    other.clear();
}

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

StringImpl* String::get_impl_ptr_unsafe() const noexcept {
    return impl_.get();
}

const ObjectPtr<StringImpl>& String::get_object_ptr() const {
    return impl_;
}

StringImpl* String::release_impl_unsafe() {
    return impl_.release();
}

String::operator std::string() const {
    return {data(), size()};
}

int String::compare(const String& other) const {
    return memncmp(data(), size(), other.data(), other.size());
}

int String::compare(const std::string& other) const {
    return memncmp(data(), size(), other.data(), other.size());
}

int String::compare(const char* other) const {
    return memncmp(data(), size(), other, std::strlen(other));
}

int String::memncmp(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
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

bool String::memequal(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
    // return (lhs_cnt == rhs_cnt) && (lhs == rhs || std::memcmp(lhs, rhs, lhs_cnt) == 0);
    return memncmp(lhs, lhs_cnt, rhs, rhs_cnt) == 0;
}

String String::concat(const char* lhs, size_t lhs_cnt, const char* rhs, size_t rhs_cnt) {
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
    return String::concat(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

String operator+(const String& lhs, const std::string& rhs) {
    return String::concat(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

String operator+(const std::string& lhs, const String& rhs) {
    return String::concat(lhs.data(), lhs.size(), rhs.data(), rhs.size());
}

String operator+(const String& lhs, const char* rhs) {
    return String::concat(lhs.data(), lhs.size(), rhs, std::strlen(rhs));
}

String operator+(const char* lhs, const String& rhs) {
    return String::concat(lhs, std::strlen(lhs), rhs.data(), rhs.size());
}

std::ostream& operator<<(std::ostream& os, const String& str) {
    os.write(str.data(), str.size());
    return os;
}


}// namespace aethermind