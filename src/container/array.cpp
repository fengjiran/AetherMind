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

void ArrayImpl::ShrinkBy(int64_t delta) {
    auto* p = end();
    while (delta > 0) {
        (--p)->~Any();
        --size_;
        --delta;
    }
}


}// namespace aethermind
