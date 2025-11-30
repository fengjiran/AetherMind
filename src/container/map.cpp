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

ObjectPtr<DenseMapImpl> DenseMapImpl::Create(uint32_t fib_shift, size_t slot_num) {
    CHECK(slot_num > SmallMapImpl::kMaxSize);
    CHECK((slot_num & kSmallMapMask) == 0ull);
    size_t block_num = ComputeBlockNum(slot_num);
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
    impl->size_ = 0;
    impl->slot_ = slot_num;
    impl->fib_shift_ = fib_shift;
    impl->iter_list_head_ = kInvalidIndex;
    impl->iter_list_tail_ = kInvalidIndex;

    auto* p = static_cast<Block*>(impl->data_);
    for (size_t i = 0; i < block_num; ++i) {
        new (p++) Block;
    }
    return impl;
}

const size_t DenseMapImpl::NextProbePosOffset[kNumJumpDists] = {
        /* clang-format off */
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      // Quadratic probing with triangle numbers. See also:
      // 1) https://en.wikipedia.org/wiki/Quadratic_probing
      // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
      // 3) https://github.com/skarupke/flat_hash_map
      21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
      136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
      351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
      666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
      1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
      1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
      2211, 2278, 2346, 2415, 2485, 2556, 2628,
      // larger triangle numbers
      8515, 19110, 42778, 96141, 216153,
      486591, 1092981, 2458653, 5532801, 12442566,
      27993903, 62983476, 141717030, 318844378, 717352503,
      1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
      93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
      5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
      309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
      17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251,
      457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435,
};

}// namespace aethermind
