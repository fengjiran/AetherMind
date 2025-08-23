//
// Created by 赵丹 on 2025/8/22.
//
#include "container/string.h"

#include <cstring>

namespace aethermind {

StringImplNullType StringImplNullType::singleton_;

StringImpl::StringImpl() : data_(nullptr), size_(0) {}

size_t StringImpl::size() const noexcept {
    return size_;
}

const char* StringImpl::data() const noexcept {
    return data_;
}

const char* String::c_str() const noexcept {
    return data();
}

String::String(const char* other, size_t size)
    : impl_(make_array_object<StringImpl, char>(size + 1)) {
    char* dst = reinterpret_cast<char*>(impl_.get()) + sizeof(StringImpl);
    impl_->data_ = dst;
    impl_->size_ = size;
    std::memcpy(dst, other, size);
    dst[size] = '\0';
}

String::String(const char* other) : String(other, strlen(other)) {}

String::String(const std::string& other) : String(other.data(), other.size()) {}

void String::swap(String& other) noexcept {
    std::swap(impl_, other.impl_);
}


const char* String::data() const noexcept {
    return impl_->data();
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

}// namespace aethermind