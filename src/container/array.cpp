//
// Created by 赵丹 on 2025/9/1.
//

#include "container/array.h"

namespace aethermind {

ArrayImpl::~ArrayImpl() {
    clear();
}

void ArrayImpl::ConstructOneElemAtEnd(Any value) {
    AM_CHECK(capacity() - size() >= 1);
    auto* p = end();
    new (p) Any(std::move(value));
    ++size_;
}

void ArrayImpl::ConstructAtEnd(size_t n, const Any& value) {
    AM_CHECK(n <= capacity() - size());
    auto* p = end();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (size_t i = 0; i < n; ++i) {
        new (p++) Any(value);
        ++size_;
    }
}

void ArrayImpl::ShrinkBy(int64_t delta) {
    AM_CHECK(delta <= size());
    for (auto* p = end(); delta > 0; --delta) {
        (--p)->~Any();
        --size_;
    }
}

void ArrayImpl::EnlargeBy(int64_t delta, const Any& value) {
    ConstructAtEnd(delta, value);
}

void ArrayImpl::clear() {
    ShrinkBy(static_cast<int64_t>(size()));
}

void ArrayImpl::MoveElemsRight(size_t dst, size_t src, size_t n) {
    AM_CHECK(src < dst);
    auto* from = begin() + src + n;
    auto* to = begin() + dst + n;
    for (size_t i = 0; i < n; ++i) {
        *--to = std::move(*--from);
    }
}

void ArrayImpl::MoveElemsLeft(size_t dst, size_t src, size_t n) {
    AM_CHECK(src > dst);
    auto* from = begin() + src;
    auto* to = begin() + dst;
    for (size_t i = 0; i < n; ++i) {
        *to++ = std::move(*from++);
    }
}

}// namespace aethermind
