//
// Created by richard on 11/28/25.
//

#include "container/map.h"

namespace aethermind {

class MapImpl::iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = KVType;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = int64_t;

    iterator() : index_(0), ptr_(nullptr) {}

    // pointer operator->() const {
    //     //
    // }

    bool operator==(const iterator& other) const {
        return index_ == other.index_ && ptr_ == other.ptr_;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

protected:
    iterator(size_t index, const MapImpl* ptr) : index_(index), ptr_(ptr) {}

    size_t index_;      // The position in the array.
    const MapImpl* ptr_;// The container it pointer to.

    friend class SmallMapImpl;
    friend class DenseMapImpl;
};

ObjectPtr<SmallMapImpl> SmallMapImpl::Create(size_t n) {
    auto impl = make_array_object<SmallMapImpl, KVType>(n);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(SmallMapImpl);
    impl->size_ = 0;
    impl->slot_ = n & ~kSmallMapMask | kSmallMapMask;
    return impl;
}

ObjectPtr<SmallMapImpl> SmallMapImpl::CopyFrom(const SmallMapImpl* src) {
    auto* first = static_cast<KVType*>(src->data_);
    auto* last = first + src->size_;
    return CreateFromRange(src->size_, first, last);
}


}// namespace aethermind
