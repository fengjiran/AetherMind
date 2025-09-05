//
// Created by 赵丹 on 2025/9/1.
//

#include "container/array.h"

namespace aethermind {

ArrayImplNullType ArrayImplNullType::singleton_;

ArrayImpl::~ArrayImpl() {
    auto* p = begin();
    for (size_t i = 0; i < size(); ++i) {
        (p + i)->~Any();
    }
}

void ArrayImpl::ConstructAtEnd(size_t n, const Any& value) {
    auto* p = end();
    // placement new
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (size_t i = 0; i < n; ++i) {
        new (p++) Any(value);
        ++size_;
    }
}

void ArrayImpl::ShrinkBy(int64_t delta) {
    auto* p = end();
    while (delta > 0) {
        (--p)->~Any();
        --size_;
        --delta;
    }
}

void ArrayImpl::EnlargeBy(int64_t delta, const Any& value) {
    auto* p = end();
    while (delta > 0) {
        new (p++) Any(value);
        ++size_;
        --delta;
    }
}

void ArrayImpl::clear() {
    ShrinkBy(size());
}

void ArrayImpl::MoveElemsRight(size_t dst, size_t src, size_t n) {
    CHECK(src < dst);
    auto* from = begin() + src + n;
    auto* to = begin() + dst + n;
    for (size_t i = 0; i < n; ++i) {
        *--to = std::move(*--from);
    }
}

void ArrayImpl::MoveElemsLeft(size_t dst, size_t src, size_t n) {
    CHECK(src > dst);
    auto* from = begin() + src;
    auto* to = begin() + dst;
    for (size_t i = 0; i < n; ++i) {
        *to++ = std::move(*from++);
    }
}


ArrayImpl* ArrayImpl::create(size_t n) {
    auto pimpl = make_array_object<ArrayImpl, Any>(n);
    pimpl->start_ = reinterpret_cast<char*>(pimpl.get()) + sizeof(ArrayImpl);
    pimpl->size_ = 0;
    pimpl->capacity_ = n;
    return pimpl.release();
}


}// namespace aethermind
