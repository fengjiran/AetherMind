//
// Created by 赵丹 on 2025/9/1.
//

#include "container/array.h"

namespace aethermind {

ArrayImplNullType ArrayImplNullType::singleton_;

template<typename T>
Array<T>::Array(size_t n, Any value) : impl_(make_array_object<ArrayImpl, Any>(n)) {
    impl_->start_ = reinterpret_cast<char*>(impl_.get()) + sizeof(ArrayImpl);
    impl_->size_ = 0;
    impl_->capacity_ = n;

    auto* p = static_cast<Any*>(impl_->start_);
    for (size_t i = 0; i < n; ++i) {
        new (p + i) Any(std::move(value));
    }
}


}// namespace aethermind
